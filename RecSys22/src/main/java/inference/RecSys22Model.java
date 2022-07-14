package inference;

import com.google.common.util.concurrent.AtomicDouble;
import com.google.common.util.concurrent.AtomicDoubleArray;
import linalg.MLVector;
import linalg.MLVectorUtils;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import model.xgboost.MLXGBUtils;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.slf4j.LoggerFactory;
import utils.MLAsync;
import utils.MLIOUtils;
import utils.MLSortUtils;
import utils.MLTimer;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.time.LocalDateTime;
import java.time.ZonedDateTime;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class RecSys22Model {

    public static MLTimer TIMER;

    static {
        MLTimer.initDefaultLogger();
        TIMER = new MLTimer(LoggerFactory.getLogger(RecSys22Model.class));
    }

    public RecSys22Data data;

    public RecSys22Model(final RecSys22Data data) {
        this.data = data;
    }

    public void extractTrainData(final RecSys22ModelConfig config) throws Exception {
        TIMER.tic();

        //slide month-long window across training set to simulate test setting
        ZonedDateTime curDateTime = LocalDateTime.parse(RecSys22Split.TRAIN_START_DATE,
                RecSys22Helper.DATE_FORMATTER).atZone(RecSys22Helper.GMT);
        final long trainEndDate = RecSys22Helper.convertToMillis(RecSys22Split.TRAIN_END_DATE);
        final long validStartDate = config.validStartDateTime.toInstant().toEpochMilli();

        try (BufferedWriter writerTrain = new BufferedWriter(
                new FileWriter(config.xgbModelPath + "TRAIN"));
             BufferedWriter writerValid = new BufferedWriter(
                     new FileWriter(config.xgbModelPath + "VALID"))) {
            while (true) {
                long startDate = curDateTime.toInstant().toEpochMilli();
                curDateTime = curDateTime.plusMonths(1);
                long endDate = curDateTime.toInstant().toEpochMilli();
                boolean isValid;
                if (config.removeValidData) {
                    isValid = (startDate >= validStartDate);
                } else {
                    isValid = false;
                }

                RecSys22FeatExtractor featExtractor = new RecSys22FeatExtractor(
                        this.data,
                        config,
                        startDate,
                        endDate);

                //extract features for all sessions that fall into the interval
                long[][] sessionDates =
                        this.data.split.splitSessionDates.get(RecSys22Split.TRAIN_SET);
                int[][] sessionItems =
                        this.data.split.splitSessionItems.get(RecSys22Split.TRAIN_SET);
                int[] sessionTargets =
                        this.data.split.splitSessionTargets.get(RecSys22Split.TRAIN_SET);

                //get all candidate items for the interval
                int[] candidates = RecSys22Helper.getAllCandidates(
                        sessionDates,
                        sessionTargets,
                        startDate,
                        endDate);

                String[][] allFeatures = new String[sessionDates.length][];
                AtomicInteger counter = new AtomicInteger();
                IntStream.range(0, sessionDates.length).parallel().forEach(index -> {
                    if (!RecSys22Helper.isInInterval(sessionDates[index], startDate, endDate)) {
                        return;
                    }
                    counter.incrementAndGet();

                    RecSys22Session session = new RecSys22Session();
                    session.dates = sessionDates[index];
                    session.items = sessionItems[index];
                    session.targetItem = sessionTargets[index];
                    if (isValid) {
                        session.candidateItems = RecSys22Helper.sampleCandidates(
                                session.targetItem,
                                session.items,
                                candidates,
                                index,
                                500);
                        RecSys22Helper.sampleSession(session, index);
                    } else {
                        session.candidateItems = RecSys22Helper.sampleCandidates(
                                session.targetItem,
                                session.items,
                                candidates,
                                index,
                                config.xgbNCandidates);
                    }

                    //compute features and targets for all candidateItems
                    List<MLVector>[] features = featExtractor.extractFeatures(session);
                    float[] targets = featExtractor.extractTargets(session);

                    //convert features to libSVM strings
                    String[] sessionFeatures = new String[features.length];
                    for (int i = 0; i < features.length; i++) {
                        sessionFeatures[i] = String.format("%d qid:%d", (int) targets[i], index)
                                + MLVectorUtils.toLibSVM(features[i], 4, true) + "\n";
                    }
                    allFeatures[index] = sessionFeatures;
                });
                TIMER.toc("extractTrainSet() [" + startDate + ", " + endDate + "] nSessions:" + counter.get() + " nCandidates:" + candidates.length + " isValid:" + isValid);

                //write libSVM sequentially to avoid randomness issues
                for (String[] sessionFeatures : allFeatures) {
                    if (sessionFeatures == null) {
                        continue;
                    }
                    if (isValid) {
                        for (String sessionFeature : sessionFeatures) {
                            writerValid.write(sessionFeature);
                        }
                    } else {
                        for (String sessionFeature : sessionFeatures) {
                            writerTrain.write(sessionFeature);
                            if (!config.removeValidData) {
                                writerValid.write(sessionFeature);
                            }
                        }
                    }
                }

                if (endDate == trainEndDate) {
                    break;
                }
            }
        }
    }

    public void train(final RecSys22ModelConfig config) throws Exception {
        TIMER.tic();
        TIMER.toc("train() starting model training");

        DMatrix dMatrixTrain = new DMatrix(config.xgbModelPath + "TRAIN");
        DMatrix dMatrixValid = new DMatrix(config.xgbModelPath + "VALID");
        TIMER.toc("train() train nRows " + dMatrixTrain.rowNum());
        TIMER.toc("valid() valid nRows " + dMatrixValid.rowNum());

        //set XGB parameters
        Map<String, Object> xgbParams = new HashMap<>();
        xgbParams.put("booster", "gbtree");
        xgbParams.put("verbosity", 1);
        xgbParams.put("eta", 0.1);
        xgbParams.put("gamma", 0);
        xgbParams.put("min_child_weight", 20);
        xgbParams.put("max_depth", 5);
        xgbParams.put("subsample", 1);
        xgbParams.put("colsample_bytree", 0.8);
        xgbParams.put("alpha", 0);
        xgbParams.put("lambda", 200);
        xgbParams.put("tree_method", "hist");
        xgbParams.put("seed", 1);
        xgbParams.put("base_score", 0.1);
        xgbParams.put("objective", "binary:logistic");
        xgbParams.put("eval_metric", "map");
        xgbParams.put("verbose_eval", 50);
        TIMER.toc("train() xbg params " + xgbParams);
        TIMER.toc("train() nRounds " + config.xgbNTrees);

        //set watches
        HashMap<String, DMatrix> watches = new HashMap<>();
        watches.put("valid", dMatrixValid);

        //train
        Booster booster = XGBoost.train(
                dMatrixTrain,
                xgbParams,
                config.xgbNTrees,
                watches,
                null,
                null);

        //save
        String modelFile = config.xgbModelPath + config.xgbNTrees + ".model";
        booster.saveModel(modelFile);
        TIMER.toc("save model " + modelFile);

        //clean-up
        dMatrixTrain.dispose();
        dMatrixValid.dispose();
        booster.dispose();
    }

    public void validate(final RecSys22ModelConfig config) {
        TIMER.tic();
        MLAsync<Booster> xgbModelFactory =
                MLXGBUtils.asyncModel(config.xgbModelPath + config.xgbFirstStageModel, 1);
        TIMER.toc("validate() " + config.xgbModelPath + config.xgbFirstStageModel);

        ZonedDateTime trainEndDateTime = LocalDateTime.parse(RecSys22Split.TRAIN_END_DATE,
                RecSys22Helper.DATE_FORMATTER).atZone(RecSys22Helper.GMT);
        ZonedDateTime validDateTime = trainEndDateTime.minusMonths(1);

        long startDate = validDateTime.toInstant().toEpochMilli();
        validDateTime = validDateTime.plusMonths(1);
        long endDate = validDateTime.toInstant().toEpochMilli();

        RecSys22FeatExtractor featExtractor = new RecSys22FeatExtractor(
                this.data,
                config,
                startDate,
                endDate);

        long[][] sessionDates =
                this.data.split.splitSessionDates.get(RecSys22Split.TRAIN_SET);
        int[][] sessionItems =
                this.data.split.splitSessionItems.get(RecSys22Split.TRAIN_SET);
        int[] sessionTargets =
                this.data.split.splitSessionTargets.get(RecSys22Split.TRAIN_SET);

        //get all candidate items for the interval
        int[] candidates = RecSys22Helper.getAllCandidates(
                sessionDates,
                sessionTargets,
                startDate,
                endDate);

        AtomicInteger counter = new AtomicInteger();
        int[] thresholds = new int[]{100, 200, 300, 400, 500};

        int[] nTrees = new int[config.xgbNTrees / 100];
        AtomicDouble[] mrr = new AtomicDouble[nTrees.length];
        AtomicDoubleArray[] recall = new AtomicDoubleArray[nTrees.length];
        for (int i = 0; i < nTrees.length; i++) {
            nTrees[i] = (i + 1) * 100;
            mrr[i] = new AtomicDouble(0);
            recall[i] = new AtomicDoubleArray(thresholds.length);
        }
        IntStream.range(0, sessionDates.length).parallel().forEach(index -> {
            if (!RecSys22Helper.isInInterval(sessionDates[index], startDate, endDate)) {
                return;
            }
            int count = counter.incrementAndGet();
            if (count % 10_000 == 0) {
                TIMER.tocLoop("validate()", count);
            }

            RecSys22Session session = new RecSys22Session();
            session.dates = sessionDates[index];
            session.items = sessionItems[index];
            session.targetItem = sessionTargets[index];
            session.candidateItems = RecSys22Helper.removeSessionItems(
                    session.items,
                    candidates);
            RecSys22Helper.sampleSession(session, index);

            List<MLVector>[] features = featExtractor.extractFeatures(session);
            float[] targets = featExtractor.extractTargets(session);
            try {
                DMatrix sessionMat = MLXGBUtils.toDMatrix(features, null, true);
                Booster model = xgbModelFactory.get();
                for (int t = 0; t < nTrees.length; t++) {
                    float[][] preds = model.predict(sessionMat, false, nTrees[t]);
                    float[] predsFlat = new float[preds.length];
                    for (int i = 0; i < preds.length; i++) {
                        predsFlat[i] = preds[i][0];
                    }
                    float[] targetsSorted = targets.clone();
                    MLSortUtils.coSort(predsFlat, targetsSorted, false);
                    for (int i = 0; i < thresholds[thresholds.length - 1]; i++) {
                        if (targetsSorted[i] == 1) {
                            if (i < 100) {
                                mrr[t].addAndGet(1.0 / (i + 1.0));
                            }
                            for (int j = 0; j < thresholds.length; j++) {
                                if (thresholds[j] >= i) {
                                    recall[t].addAndGet(j, 1.0);
                                }
                            }
                            break;
                        }
                    }
                }
                sessionMat.dispose();
            } catch (Exception e) {
                e.printStackTrace();
                throw new RuntimeException(e);
            }
        });

        for (int t = 0; t < nTrees.length; t++) {
            TIMER.toc("nTrees:" + nTrees[t]
                    + " MRR:" + String.format("%.4f", mrr[t].get() / counter.get()));
            StringBuilder builder = new StringBuilder();
            builder.append("nTrees:" + nTrees[t] + " RECALL");
            for (int i = 0; i < thresholds.length; i++) {
                builder.append("  R@" + thresholds[i] + ":" +
                        String.format("%.4f", recall[t].get(i) / counter.get()));
            }
            TIMER.toc(builder.toString());
        }
    }

    public void generateSubmission(final RecSys22ModelConfig config,
                                   final String set) throws Exception {
        TIMER.tic();
        MLAsync<Booster> xgbModelFactory =
                MLXGBUtils.asyncModel(config.xgbModelPath + config.xgbFirstStageModel, 1);
        TIMER.toc("generateSubmission() " + config.xgbModelPath + config.xgbFirstStageModel + " " +
                "nTrees:" + config.xgbNTrees);

        Map<Integer, Integer> indexToItem = new HashMap<>();
        for (Map.Entry<Integer, Integer> entry : this.data.itemToIndex.entrySet()) {
            indexToItem.put(entry.getValue(), entry.getKey());
        }

        int[] sessionIds =
                this.data.split.splitSessionIds.get(set);
        long[][] sessionDates =
                this.data.split.splitSessionDates.get(set);
        int[][] sessionItems =
                this.data.split.splitSessionItems.get(set);

        int[] candidates = this.data.split.splitCandidates.get(set);
        RecSys22FeatExtractor featExtractor = new RecSys22FeatExtractor(
                this.data,
                config,
                null,
                null);
        AtomicInteger counter = new AtomicInteger();
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(
                config.xgbModelPath + "submission_" + set));
             BufferedWriter writerScores = new BufferedWriter(new FileWriter(
                     config.xgbModelPath + set
                             + "_" + config.xgbFirstStageModel
                             + "_" + config.xgbNTrees
                             + "_scores"))) {
            writer.write("session_id,item_id,rank\n");
            IntStream.range(0, sessionDates.length).parallel().forEach(index -> {
                int count = counter.incrementAndGet();
                if (count % 10_000 == 0) {
                    TIMER.tocLoop("generateSubmission() " + set, count, sessionDates.length);
                }
                RecSys22Session session = new RecSys22Session();
                session.dates = sessionDates[index];
                session.items = sessionItems[index];
                session.targetItem = -1;
                session.candidateItems = candidates.clone();

                int[] itemsSorted = session.items.clone();
                Arrays.sort(itemsSorted);

                List<MLVector>[] features = featExtractor.extractFeatures(session);
                try {
                    DMatrix sessionMat = MLXGBUtils.toDMatrix(features, null, true);
                    Booster model = xgbModelFactory.get();
                    float[][] preds = model.predict(sessionMat, false, config.xgbNTrees);
                    float[] predsFlat = new float[preds.length];
                    for (int i = 0; i < preds.length; i++) {
                        predsFlat[i] = preds[i][0];
                    }
                    MLSortUtils.coSort(predsFlat, session.candidateItems, false);
                    int sessionId = sessionIds[index];

                    StringBuilder builderTop = new StringBuilder();
                    StringBuilder builderAll = new StringBuilder();
                    int topCount = 0;
                    for (int i = 0; i < predsFlat.length; i++) {
                        if (topCount < 100 &&
                                Arrays.binarySearch(itemsSorted, session.candidateItems[i]) < 0) {
                            builderTop.append(sessionId).append(",").append(indexToItem.get(session.candidateItems[i])).append(",").append(topCount + 1).append("\n");
                            topCount++;
                        }

                        if (Arrays.binarySearch(itemsSorted, session.candidateItems[i]) < 0) {
                            builderAll.append(sessionId).append(",").append(indexToItem.get(session.candidateItems[i])).append(",").append(String.format("%.4f", predsFlat[i])).append("\n");
                        } else {
                            builderAll.append(sessionId).append(",").append(indexToItem.get(session.candidateItems[i])).append(",").append("-inf").append("\n");
                        }
                    }
                    synchronized (this) {
                        writer.write(builderTop.toString());
                        writerScores.write(builderAll.toString());
                    }

                    sessionMat.dispose();
                } catch (Exception e) {
                    e.printStackTrace();
                    throw new RuntimeException(e);
                }
            });
        }
    }


    public static RecSys22ModelConfig loadConfig(final CommandLine cmd) {
        RecSys22ModelConfig modelConfig = new RecSys22ModelConfig();
        modelConfig.dataPath = cmd.getOptionValue("dataPath");

        modelConfig.xgbModelPath = cmd.getOptionValue("xgbModelPath");
        if (modelConfig.xgbModelPath != null && !(new File(modelConfig.xgbModelPath).exists())) {
            //create data folder if needed
            new File(modelConfig.dataPath).mkdir();
        }

        modelConfig.xgbNCandidates = Integer.parseInt(
                cmd.getOptionValue("xgbNCandidates") == null ? "0"
                        : cmd.getOptionValue("xgbNCandidates"));

        modelConfig.xgbNTrees = Integer.parseInt(
                cmd.getOptionValue("xgbNTrees") == null ? "0"
                        : cmd.getOptionValue("xgbNTrees"));

        modelConfig.xgbFirstStageModel = cmd.getOptionValue("xgbFirstStageModel");

        return modelConfig;
    }

    public static void main(String[] args) {
        try {
            RecSys22ModelOptions cmdOptions = new RecSys22ModelOptions();
            CommandLineParser parser = new DefaultParser();
            HelpFormatter formatter = new HelpFormatter();
            CommandLine cmd = null;
            try {
                cmd = parser.parse(cmdOptions.options, args);
            } catch (Exception e) {
                e.printStackTrace();
                formatter.printHelp(" ", cmdOptions.options);
                System.exit(1);
            }

            RecSys22ModelConfig config = loadConfig(cmd);
            ZonedDateTime trainEndDateTime = LocalDateTime.parse(RecSys22Split.TRAIN_END_DATE,
                    RecSys22Helper.DATE_FORMATTER).atZone(RecSys22Helper.GMT);

            config.validStartDateTime = trainEndDateTime.minusMonths(1);
            config.removeValidData = false;

            if (cmd.getOptionValue("action").equals("extractFeatures")) {
                RecSys22Data data = MLIOUtils.readObjectFromFile(
                        config.dataPath + "RecSys22Data",
                        RecSys22Data.class);
                RecSys22Model model = new RecSys22Model(data);
                model.extractTrainData(config);

            } else if (cmd.getOptionValue("action").equals("trainModel")) {
                RecSys22Data data = MLIOUtils.readObjectFromFile(
                        config.dataPath + "RecSys22Data",
                        RecSys22Data.class);
                RecSys22Model model = new RecSys22Model(data);
                model.train(config);
                if (config.removeValidData) {
                    model.validate(config);
                }

            } else if (cmd.getOptionValue("action").equals("genSubmission")) {
                RecSys22Data data = MLIOUtils.readObjectFromFile(
                        config.dataPath + "RecSys22Data",
                        RecSys22Data.class);
                RecSys22Model model = new RecSys22Model(data);
                if (config.removeValidData) {
                    model.generateSubmission(config, RecSys22Split.VALID_SET);
                    model.generateSubmission(config, RecSys22Split.TEST_LB_SET);
                } else {
                    model.generateSubmission(config, RecSys22Split.TEST_LB_SET);
                    model.generateSubmission(config, RecSys22Split.TEST_FINAL_SET);
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}