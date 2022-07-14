package inference;

import linalg.MLMatrixSparseMultiColFloat;
import org.slf4j.LoggerFactory;
import utils.MLSortUtils;
import utils.MLTimer;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.stream.IntStream;

public class RecSys22CF {

    public static MLTimer TIMER;

    static {
        MLTimer.initDefaultLogger();
        TIMER = new MLTimer(LoggerFactory.getLogger(RecSys22CF.class));
    }

    public RecSys22Data data;
    public RecSys22ModelConfig config;

    public MLMatrixSparseMultiColFloat itemItemBuy;
    public MLMatrixSparseMultiColFloat itemItemSession;

    public RecSys22CF(final RecSys22Data data,
                      final RecSys22ModelConfig config,
                      final Long targetStartDate,
                      final Long targetEndDate) {
        this.data = data;
        this.config = config;
        this.initItemItemBuy(targetStartDate, targetEndDate);
        this.initItemItemSession(targetStartDate, targetEndDate);
    }

    public void initItemItemSession(final Long targetStartDate, final Long targetEndDate) {
        TIMER.tic();
        int N_ITEMS = this.data.itemToIndex.size();

        long[][] sessionDates =
                this.data.split.splitSessionDates.get(RecSys22Split.TRAIN_SET);
        int[][] sessionItems =
                this.data.split.splitSessionItems.get(RecSys22Split.TRAIN_SET);

        ConcurrentMap<Integer, AtomicInteger>[] maps = new ConcurrentHashMap[N_ITEMS];
        for (int i = 0; i < maps.length; i++) {
            maps[i] = new ConcurrentHashMap<>();
        }

        long validStartDate = this.config.validStartDateTime.toInstant().toEpochMilli();
        AtomicIntegerArray itemSessionCount = new AtomicIntegerArray(N_ITEMS);
        IntStream.range(0, sessionDates.length).parallel().forEach(index -> {
            if (targetStartDate != null && RecSys22Helper.isInInterval(sessionDates[index],
                    targetStartDate, targetEndDate)) {
                return;
            }
            if (this.config.removeValidData && sessionDates[index][0] >= validStartDate) {
                return;
            }

            int[] items = sessionItems[index];
            Set<Integer> uniqueItems = new HashSet<>();
            for (int item : items) {
                uniqueItems.add(item);
            }

            for (int item1 : uniqueItems) {
                itemSessionCount.incrementAndGet(item1);
                for (int item2 : uniqueItems) {
                    if (item1 == item2) {
                        continue;
                    }
                    AtomicInteger count = maps[item1].computeIfAbsent(
                            item2,
                            k -> new AtomicInteger(0));
                    count.getAndIncrement();
                }
            }
        });
        int[][] indexes = new int[N_ITEMS][];
        float[][] values = new float[N_ITEMS][];
        IntStream.range(0, N_ITEMS).parallel().forEach(itemIndex -> {
            Map<Integer, AtomicInteger> map = maps[itemIndex];
            if (map.size() == 0) {
                return;
            }
            int[] curIndexes = new int[map.size()];
            float[] curValues = new float[map.size()];
            int cur = 0;
            for (Map.Entry<Integer, AtomicInteger> entry : map.entrySet()) {
                int item = entry.getKey();
                double count = entry.getValue().get();
                count = count / (Math.sqrt(Math.max(itemSessionCount.get(itemIndex), 1.0)) *
                        Math.sqrt(Math.max(itemSessionCount.get(item), 1.0)));

                curIndexes[cur] = item;
                curValues[cur] = (float) count;
                cur++;
            }
            MLSortUtils.coSort(curIndexes, curValues, true);
            indexes[itemIndex] = curIndexes;
            values[itemIndex] = curValues;
        });
        this.itemItemSession = new MLMatrixSparseMultiColFloat(indexes, values, N_ITEMS);
        TIMER.toc("initItemItemSession() done");
    }

    public void initItemItemBuy(final Long targetStartDate, final Long targetEndDate) {
        TIMER.tic();
        int N_ITEMS = this.data.itemToIndex.size();

        long[][] sessionDates =
                this.data.split.splitSessionDates.get(RecSys22Split.TRAIN_SET);
        int[][] sessionItems =
                this.data.split.splitSessionItems.get(RecSys22Split.TRAIN_SET);
        int[] sessionTargets =
                this.data.split.splitSessionTargets.get(RecSys22Split.TRAIN_SET);

        ConcurrentMap<Integer, AtomicInteger>[] maps = new ConcurrentHashMap[N_ITEMS];
        for (int i = 0; i < maps.length; i++) {
            maps[i] = new ConcurrentHashMap<>();
        }

        long validStartDate = this.config.validStartDateTime.toInstant().toEpochMilli();
        AtomicIntegerArray itemBuyCount = new AtomicIntegerArray(N_ITEMS);
        AtomicIntegerArray itemSessionCount = new AtomicIntegerArray(N_ITEMS);
        IntStream.range(0, sessionDates.length).parallel().forEach(index -> {
            if (targetStartDate != null && RecSys22Helper.isInInterval(sessionDates[index],
                    targetStartDate, targetEndDate)) {
                return;
            }
            if (this.config.removeValidData && sessionDates[index][0] >= validStartDate) {
                return;
            }

            int[] items = sessionItems[index];
            int targetItem = sessionTargets[index];
            itemBuyCount.incrementAndGet(targetItem);
            Set<Integer> uniqueItems = new HashSet<>();
            for (int item : items) {
                uniqueItems.add(item);
            }

            ConcurrentMap<Integer, AtomicInteger> map = maps[targetItem];
            for (int item : uniqueItems) {
                itemSessionCount.incrementAndGet(item);
                AtomicInteger count = map.computeIfAbsent(
                        item,
                        k -> new AtomicInteger(0));
                count.getAndIncrement();
            }
        });
        int[][] indexes = new int[N_ITEMS][];
        float[][] values = new float[N_ITEMS][];
        IntStream.range(0, N_ITEMS).parallel().forEach(itemIndex -> {
            Map<Integer, AtomicInteger> map = maps[itemIndex];
            if (map.size() == 0) {
                return;
            }
            int[] curIndexes = new int[map.size()];
            float[] curValues = new float[map.size()];
            int cur = 0;
            for (Map.Entry<Integer, AtomicInteger> entry : map.entrySet()) {
                int item = entry.getKey();
                double count = entry.getValue().get();
                count = count / (Math.sqrt(Math.max(itemBuyCount.get(itemIndex), 1.0)) *
                        Math.sqrt(Math.max(itemSessionCount.get(item), 1.0)));

                curIndexes[cur] = item;
                curValues[cur] = (float) count;
                cur++;
            }
            MLSortUtils.coSort(curIndexes, curValues, true);
            indexes[itemIndex] = curIndexes;
            values[itemIndex] = curValues;
        });
        this.itemItemBuy = new MLMatrixSparseMultiColFloat(indexes, values, N_ITEMS);
        TIMER.toc("initItemItemBuy() done");
    }

    public static float getCFScore(final int targetItem,
                                   final int sessionItem,
                                   final MLMatrixSparseMultiColFloat R) {
        int[] indexes = R.indexes[targetItem];
        if (indexes == null) {
            return 0f;
        }
        int index = Arrays.binarySearch(indexes, sessionItem);
        if (index < 0) {
            return 0f;
        }
        return R.values[targetItem][index];
    }
}
