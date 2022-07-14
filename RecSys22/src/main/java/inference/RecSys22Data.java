package inference;

import feature.MLFeatureDataCatMultiColFloat;
import feature.MLFeatureDerivedColSelect;
import org.slf4j.LoggerFactory;
import utils.MLIOUtils;
import utils.MLSortUtils;
import utils.MLTimer;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class RecSys22Data implements Serializable {

    private static final long serialVersionUID = -5172273246919569913L;
    public static MLTimer TIMER;

    static {
        MLTimer.initDefaultLogger();
        TIMER = new MLTimer(LoggerFactory.getLogger(RecSys22Data.class));
    }

    public Map<Integer, Integer> itemToIndex;
    public RecSys22Split split;

    public int[][] itemCatIds;
    public int[][] itemCatVals;

    public MLFeatureDataCatMultiColFloat itemCatFeature;
    public MLFeatureDerivedColSelect itemCatColSelect;

    public MLFeatureDataCatMultiColFloat itemCatValFeature;
    public MLFeatureDerivedColSelect itemCatValColSelect;

    public RecSys22Data() {

    }

    public void loadSessions(final String file, final String set) throws Exception {
        TIMER.tic();
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            int[] sessionIds = new int[1_000_000];
            int[][] sessionItems = new int[1_000_000][];
            long[][] sessionDates = new long[1_000_000][];

            List<Integer> curItems = new ArrayList<>();
            List<Long> curDates = new ArrayList<>();
            int sessionCount = 0;
            String line;
            reader.readLine();
            int curSessionId = -1;
            while ((line = reader.readLine()) != null) {
                //format: session_id,item_id,date
                String[] split = line.split(",");
                int sessionId = Integer.parseInt(split[0]);
                int itemIndex = this.itemToIndex.computeIfAbsent(
                        Integer.parseInt(split[1]),
                        k -> this.itemToIndex.size());
                long date = RecSys22Helper.convertToMillis(split[2]);

                if (curSessionId < 0 || curSessionId == sessionId) {
                    if (curSessionId < 0) {
                        curSessionId = sessionId;
                    }
                } else {
                    int[] items = curItems.stream().mapToInt(Integer::intValue).toArray();
                    long[] dates = curDates.stream().mapToLong(Long::longValue).toArray();
                    MLSortUtils.coSort(dates, items, true);
                    sessionIds[sessionCount] = curSessionId;
                    sessionItems[sessionCount] = items;
                    sessionDates[sessionCount] = dates;

                    curSessionId = sessionId;
                    curItems = new ArrayList<>();
                    curDates = new ArrayList<>();
                    sessionCount++;
                }
                curItems.add(itemIndex);
                curDates.add(date);
            }
            //last session
            int[] items = curItems.stream().mapToInt(Integer::intValue).toArray();
            long[] dates = curDates.stream().mapToLong(Long::longValue).toArray();
            MLSortUtils.coSort(dates, items, true);
            sessionIds[sessionCount] = curSessionId;
            sessionItems[sessionCount] = items;
            sessionDates[sessionCount] = dates;
            sessionCount++;

            sessionIds = Arrays.copyOfRange(sessionIds, 0, sessionCount);
            this.split.splitSessionIds.put(set, sessionIds);
            sessionItems = Arrays.copyOfRange(sessionItems, 0, sessionCount);
            this.split.splitSessionItems.put(set, sessionItems);
            sessionDates = Arrays.copyOfRange(sessionDates, 0, sessionCount);
            this.split.splitSessionDates.put(set, sessionDates);
            TIMER.toc(set + " sessions " + sessionCount);
        }
    }

    public void loadTargets(final String file) throws Exception {
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            reader.readLine();
            int[] targets =
                    new int[this.split.splitSessionItems.get(RecSys22Split.TRAIN_SET).length];
            int sessionCount = 0;
            while ((line = reader.readLine()) != null) {
                //format: session_id,item_id,date
                String[] split = line.split(",");
                int itemIndex = this.itemToIndex.computeIfAbsent(
                        Integer.parseInt(split[1]),
                        k -> this.itemToIndex.size());
                targets[sessionCount] = itemIndex;
                sessionCount++;
            }
            this.split.splitSessionTargets.put(RecSys22Split.TRAIN_SET, targets);
        }
    }

    public void loadCandidates(final String file) throws Exception {
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            reader.readLine();
            int[] candidates = new int[10_000];
            int counter = 0;
            while ((line = reader.readLine()) != null) {
                //format: item_id
                int itemIndex = this.itemToIndex.computeIfAbsent(
                        Integer.parseInt(line),
                        k -> this.itemToIndex.size());
                candidates[counter] = itemIndex;
                counter++;
            }
            candidates = Arrays.copyOfRange(candidates, 0, counter);
            this.split.splitCandidates.put(RecSys22Split.TEST_LB_SET, candidates);
            this.split.splitCandidates.put(RecSys22Split.TEST_FINAL_SET, candidates);
            TIMER.toc(RecSys22Split.TEST_LB_SET + " candidates " + counter);
        }
    }

    public void loadItemFeatures(final String file) throws Exception {
        TIMER.tic();
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            this.itemCatIds = new int[this.itemToIndex.size()][];
            this.itemCatVals = new int[this.itemToIndex.size()][];

            List<Integer> curIds = new ArrayList<>();
            List<Integer> curVals = new ArrayList<>();
            String line;
            reader.readLine();
            int currentItem = -1;
            while ((line = reader.readLine()) != null) {
                //format: item_id,feature_category_id,feature_value_id
                String[] split = line.split(",");
                int itemIndex = this.itemToIndex.get(Integer.parseInt(split[0]));
                int itemCatId = Integer.parseInt(split[1]);
                int itemCatVal = Integer.parseInt(split[2]);

                if (currentItem < 0 || currentItem == itemIndex) {
                    if (currentItem < 0) {
                        currentItem = itemIndex;
                    }
                } else {
                    int[] itemIds = curIds.stream().mapToInt(Integer::intValue).toArray();
                    int[] itemVals = curVals.stream().mapToInt(Integer::intValue).toArray();
                    MLSortUtils.coSort(itemIds, itemVals, true);
                    this.itemCatIds[currentItem] = itemIds;
                    this.itemCatVals[currentItem] = itemVals;

                    currentItem = itemIndex;
                    curIds = new ArrayList<>();
                    curVals = new ArrayList<>();
                }
                curIds.add(itemCatId);
                curVals.add(itemCatVal);
            }
            //last item
            int[] itemIds = curIds.stream().mapToInt(Integer::intValue).toArray();
            int[] itemVals = curVals.stream().mapToInt(Integer::intValue).toArray();
            MLSortUtils.coSort(itemIds, itemVals, true);
            this.itemCatIds[currentItem] = itemIds;
            this.itemCatVals[currentItem] = itemVals;
        }
        TIMER.toc("loadItemFeatures() done");
    }

    public void computeItemCatFeatures() {
        TIMER.tic();
        this.itemCatFeature = new MLFeatureDataCatMultiColFloat(this.itemCatIds.length);
        this.itemCatValFeature = new MLFeatureDataCatMultiColFloat(this.itemCatIds.length);
        IntStream.range(0, this.itemCatIds.length).forEach(index -> {
            int[] cats = this.itemCatIds[index];
            int[] vals = this.itemCatVals[index];

            String[] catStr = new String[cats.length];
            String[] catValStr = new String[cats.length];
            for (int i = 0; i < cats.length; i++) {
                catStr[i] = cats[i] + "";
                catValStr[i] = cats[i] + "-" + vals[i];
            }
            this.itemCatFeature.addRow(index, catStr);
            this.itemCatValFeature.addRow(index, catValStr);
        });
        this.itemCatFeature.finalizeFeature();
        this.itemCatColSelect =
                new MLFeatureDerivedColSelect(
                        (int) (this.itemCatIds.length * 0.05),
                        this.itemCatFeature);
        this.itemCatColSelect.computeFeature(null, null);
        TIMER.toc("ITEM CAT nCols:" + this.itemCatColSelect.getNCols());

        this.itemCatValFeature.finalizeFeature();
        this.itemCatValColSelect =
                new MLFeatureDerivedColSelect(
                        (int) (this.itemCatIds.length * 0.05),
                        this.itemCatValFeature);
        this.itemCatValColSelect.computeFeature(null, null);
        TIMER.toc("ITEM CAT-VAL nCols:" + this.itemCatValColSelect.getNCols());

        TIMER.toc("computeItemCatFeatures() done");
    }

    public void loadValidSet(final String file) throws Exception {
        TIMER.tic();
        final int N_SESSIONS = 81_618;
        int[] sessionIds = new int[N_SESSIONS];
        int[] targets = new int[N_SESSIONS];
        long[][] sessionDates = new long[N_SESSIONS][];
        int[][] sessionItems = new int[N_SESSIONS][];

        int[] trainSessionIds = this.split.splitSessionIds.get(RecSys22Split.TRAIN_SET);
        long[][] trainSessionDates = this.split.splitSessionDates.get(RecSys22Split.TRAIN_SET);

        AtomicInteger counter = new AtomicInteger(0);
        Files.lines(Paths.get(file)).parallel().forEach(line -> {
            line = line.substring(1, line.length() - 1);
            String[] split = line.split(",\\s+");

            int index = counter.getAndIncrement();
            int sessionId = Integer.parseInt(split[0]);

            int[] items = new int[split.length - 2];
            for (int i = 1; i < split.length - 1; i++) {
                items[i - 1] = this.itemToIndex.get(Integer.parseInt(split[i]));
            }

            long[] dates = null;
            for (int i = 0; i < trainSessionIds.length; i++) {
                if (trainSessionIds[i] == sessionId) {
                    dates = Arrays.copyOfRange(trainSessionDates[i], 0, items.length);
                    break;
                }
            }
            if (dates == null) {
                throw new RuntimeException("session not found");
            }

            sessionIds[index] = sessionId;
            targets[index] = this.itemToIndex.get(Integer.parseInt(split[split.length - 1]));
            sessionDates[index] = dates;
            sessionItems[index] = items;
        });
        if (counter.get() != N_SESSIONS) {
            throw new RuntimeException("wrong number of sessions");
        }

        Set<Integer> set = new HashSet<>();
        for (int target : targets) {
            set.add(target);
        }
        int[] candidates = set.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(candidates);

        this.split.splitSessionIds.put(RecSys22Split.VALID_SET, sessionIds);
        this.split.splitSessionItems.put(RecSys22Split.VALID_SET, sessionItems);
        this.split.splitSessionDates.put(RecSys22Split.VALID_SET, sessionDates);
        this.split.splitSessionTargets.put(RecSys22Split.VALID_SET, targets);
        this.split.splitCandidates.put(RecSys22Split.VALID_SET, candidates);
        TIMER.toc("loadValidSet() done");
    }

    public void loadData(final String path) throws Exception {
        this.itemToIndex = new HashMap<>();
        this.split = new RecSys22Split();

        this.loadSessions(path + "train_sessions.csv", RecSys22Split.TRAIN_SET);
        this.loadTargets(path + "train_purchases.csv");

        this.loadSessions(path + "test_leaderboard_sessions.csv", RecSys22Split.TEST_LB_SET);
        this.loadSessions(path + "test_final_sessions.csv", RecSys22Split.TEST_FINAL_SET);
        this.loadCandidates(path + "candidate_items.csv");

        this.loadItemFeatures(path + "item_features.csv");
        this.computeItemCatFeatures();
        this.loadValidSet(path + "val_sess.csv");
    }

    public static void main(final String[] args) {
        try {
            RecSys22Data data = new RecSys22Data();
            data.loadData(args[0]);

            MLIOUtils.writeObjectToFile(data, args[0] + "RecSys22Data");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
