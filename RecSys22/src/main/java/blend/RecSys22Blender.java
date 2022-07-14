package blend;

import org.slf4j.LoggerFactory;
import utils.MLIOUtils;
import utils.MLSortUtils;
import utils.MLTimer;
import inference.RecSys22Data;
import inference.RecSys22Split;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.stream.IntStream;

public class RecSys22Blender {

    public static MLTimer TIMER;

    static {
        MLTimer.initDefaultLogger();
        TIMER = new MLTimer(LoggerFactory.getLogger(RecSys22Blender.class));
    }

    public static final String[] LB_FILES = new String[]{
            "/data/recsys2022/scores/leaderboard_raw_scores.csv",

            "/data/recsys2022/scores/sigmoid_lb_score.csv",

            "/data/recsys2022/scores/jianing/jianing_lb_scores.csv",

            "/data/recsys2022/scores/zhaolin/zhaolin_lb_score_v1.csv",
            "/data/recsys2022/scores/zhaolin/zhaolin_lb_score_v2.csv",

            "/data/recsys2022/model/xgb/scores/TestLBSet_2000.model_1200_scores"
    };

    public static final String[] FINAL_FILES = new String[]{
            "/data/recsys2022/scores/final_raw_scores.csv",

            "/data/recsys2022/scores/sigmoid_final_score.csv",

            "/data/recsys2022/scores/jianing/jianing_final_scores.csv",

            "/data/recsys2022/scores/zhaolin/zhaolin_final_score_v1.csv",
            "/data/recsys2022/scores/zhaolin/zhaolin_final_score_v2.csv",

            "/data/recsys2022/model/xgb/scores/TestFinalSet_2000.model_1200_scores"
    };

    public RecSys22Data data;
    public Map<Integer, Integer> sessionToIndex;

    public int[] sessionIds;
    public int[][] sessionItems;
    public long[][] sessionDates;
    public int[] sessionTargets;
    public int[] candidates;

    public double[][][] itemScores;

    public RecSys22Blender(final RecSys22Data data,
                           final String set) {
        this.data = data;
        this.sessionIds = this.data.split.splitSessionIds.get(set);
        this.sessionItems = this.data.split.splitSessionItems.get(set);
        this.sessionDates = this.data.split.splitSessionDates.get(set);
        this.sessionTargets = this.data.split.splitSessionTargets.get(set);
        this.candidates = this.data.split.splitCandidates.get(set);
        Arrays.sort(this.candidates);

        this.sessionToIndex = new HashMap<>();
        for (int i = 0; i < this.sessionIds.length; i++) {
            this.sessionToIndex.put(this.sessionIds[i], i);
        }
    }

    public void loadScores(final String[] files) throws Exception {
        TIMER.tic();

        this.itemScores =
                new double[files.length][this.sessionToIndex.size()][this.candidates.length];
        int[][][] itemIds =
                new int[files.length][this.sessionToIndex.size()][this.candidates.length];
        for (int i = 0; i < files.length; i++) {
            AtomicIntegerArray counters = new AtomicIntegerArray(this.sessionToIndex.size());
            double[][] itemScoresFile = this.itemScores[i];
            int[][] itemIdsFile = itemIds[i];

            Files.lines(Paths.get(files[i])).parallel().forEach(line -> {
                if (line.startsWith("session_id")) {
                    return;
                }

                String[] split = line.split(",");
                int sessionIndex = this.sessionToIndex.get(Integer.parseInt(split[0]));
                int itemIndex = counters.getAndIncrement(sessionIndex);

                itemIdsFile[sessionIndex][itemIndex] =
                        this.data.itemToIndex.get(Integer.parseInt(split[1]));
                if (split[2].equals("-inf")) {
                    itemScoresFile[sessionIndex][itemIndex] = Double.NEGATIVE_INFINITY;
                } else {
                    itemScoresFile[sessionIndex][itemIndex] = Double.parseDouble(split[2]);
                }

                if (itemIndex == (this.candidates.length - 1)) {
                    MLSortUtils.coSort(
                            itemIdsFile[sessionIndex],
                            itemScoresFile[sessionIndex],
                            true);
                }
            });
            for (int j = 0; j < counters.length(); j++) {
                if (counters.get(j) != this.candidates.length) {
                    throw new Exception("incorrect number of items");
                }
            }
            TIMER.toc(files[i] + " parsed");
        }
    }

    public void normalizeScores() {
        TIMER.tic();
        for (double[][] curScores : this.itemScores) {
            IntStream.range(0, curScores.length).parallel().forEach(index -> {
                double[] scores = curScores[index];
                int[] items = this.sessionItems[index].clone();
                Arrays.sort(items);

                double mean = 0;
                double std = 0;
                double min = Double.POSITIVE_INFINITY;
                double max = Double.NEGATIVE_INFINITY;

                //mean
                int count = 0;
                for (int i = 0; i < scores.length; i++) {
                    if (Arrays.binarySearch(items, this.candidates[i]) >= 0) {
                        continue;
                    }
                    double score = scores[i];
                    count++;

                    mean += score;
                    if (min > score) {
                        min = score;
                    }
                    if (max < score) {
                        max = score;
                    }
                }
                mean = mean / count;

                //std
                for (int i = 0; i < scores.length; i++) {
                    if (Arrays.binarySearch(items, this.candidates[i]) >= 0) {
                        continue;
                    }
                    double score = scores[i];
                    std += (mean - score) * (mean - score);
                }
                std = Math.sqrt(std / count);

                //normalize
                for (int i = 0; i < scores.length; i++) {
                    if (Arrays.binarySearch(items, this.candidates[i]) >= 0) {
                        continue;
                    }
//                    scores[i] = (float) ((scores[i] - mean) / std);
                    scores[i] = (scores[i] - min) / (max - min);
                }
            });
        }
        TIMER.toc("normalize() done");
    }

    public void blend(final double[] weights,
                      final String outFile) throws Exception {
        TIMER.tic();

        Map<Integer, Integer> indexToItem = new HashMap<>();
        for (Map.Entry<Integer, Integer> entry : this.data.itemToIndex.entrySet()) {
            indexToItem.put(entry.getValue(), entry.getKey());
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outFile))) {
            writer.write("session_id,item_id,rank\n");
            IntStream.range(0, this.sessionIds.length).parallel().forEach(session -> {
                double[] blend = new double[this.candidates.length];
                for (int i = 0; i < this.itemScores.length; i++) {
                    double[] sessionScores = this.itemScores[i][session];
                    for (int j = 0; j < blend.length; j++) {
                        if (weights != null) {
                            blend[j] += weights[i] * sessionScores[j];
                        } else {
                            blend[j] += sessionScores[j];
                        }
                    }
                }
                int[] itemCandidatesClone = this.candidates.clone();
                MLSortUtils.coSort(blend, itemCandidatesClone, false);

                int[] items = this.sessionItems[session].clone();
                Arrays.sort(items);

                StringBuilder builderTop = new StringBuilder();
                int sessionId = this.sessionIds[session];
                int count = 0;
                for (int itemCandidate : itemCandidatesClone) {
                    if (Arrays.binarySearch(items, itemCandidate) >= 0) {
                        continue;
                    }
                    count++;
                    builderTop.append(sessionId).append(",").append(indexToItem.get(itemCandidate)).append(",").append(count).append("\n");
                    if (count == 100) {
                        break;
                    }
                }
                try {
                    writer.write(builderTop.toString());
                } catch (Exception e) {
                    e.printStackTrace();
                    throw new RuntimeException(e);
                }
            });
        }
        TIMER.toc("blend() done");
    }


    public static void main(final String[] args) {
        try {
            String dataPath = "/data/recsys2022/data/";
            RecSys22Data data = MLIOUtils.readObjectFromFile(
                    dataPath + "RecSys22Data",
                    RecSys22Data.class);

            RecSys22Blender blender = new RecSys22Blender(data, RecSys22Split.TEST_LB_SET);
            blender.loadScores(LB_FILES);
            blender.normalizeScores();
            blender.blend(new double[]{1, 0.50, 0.08, 0.08, 0.08, 0.08},
                    "/data/recsys2022/scores/blend/submission_blend_lb");

            blender = new RecSys22Blender(data, RecSys22Split.TEST_FINAL_SET);
            blender.loadScores(FINAL_FILES);
            blender.normalizeScores();
            blender.blend(new double[]{1, 0.50, 0.08, 0.08, 0.08, 0.08},
                    "/data/recsys2022/scores/blend/submission_blend_final");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
