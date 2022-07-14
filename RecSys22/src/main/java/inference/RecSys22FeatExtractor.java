package inference;

import linalg.MLVector;
import linalg.MLVectorDenseFloat;
import org.slf4j.LoggerFactory;
import utils.MLTimer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.IntStream;

public class RecSys22FeatExtractor {

    public static MLTimer TIMER;

    static {
        MLTimer.initDefaultLogger();
        TIMER = new MLTimer(LoggerFactory.getLogger(RecSys22FeatExtractor.class));
    }

    public RecSys22Data data;
    public RecSys22ModelConfig config;
    public RecSys22CF cf;

    public static final int SESSION_COUNT = 0;
    public static final int AVG_SESS_LEN = 1;
    public static final int AVG_SESS_DUR = 2;
    public static final int BUY_COUNT = 3;
    public static final int AVG_BOUGHT_SESS_LEN = 4;
    public static final int AVG_BOUGHT_SESS_DUR = 5;
    public static final long TRAIN_END_DATE = RecSys22Helper.convertToMillis("2021-06-01 00:00:00");
    public double[][] itemCache;
    public long[][] itemItemSessionDates;

    public RecSys22FeatExtractor(final RecSys22Data data,
                                 final RecSys22ModelConfig config,
                                 final Long targetStartDate,
                                 final Long targetEndDate) {
        this.data = data;
        this.config = config;
        this.initCache(targetStartDate, targetEndDate);
        this.cf = new RecSys22CF(this.data, this.config, targetStartDate, targetEndDate);
    }

    public void initCache(final Long targetStartDate, final Long targetEndDate) {
        TIMER.tic();
        final int N_ITEMS = this.data.itemToIndex.size();
        this.itemCache = new double[N_ITEMS][6];
        this.itemItemSessionDates = new long[N_ITEMS][N_ITEMS];

        long[][] sessionDates =
                this.data.split.splitSessionDates.get(RecSys22Split.TRAIN_SET);
        int[][] sessionItems =
                this.data.split.splitSessionItems.get(RecSys22Split.TRAIN_SET);
        int[] sessionTargets =
                this.data.split.splitSessionTargets.get(RecSys22Split.TRAIN_SET);

        long validStartDate = this.config.validStartDateTime.toInstant().toEpochMilli();
        IntStream.range(0, sessionDates.length).parallel().forEach(index -> {
            if (targetStartDate != null && RecSys22Helper.isInInterval(sessionDates[index],
                    targetStartDate, targetEndDate)) {
                return;
            }
            if (this.config.removeValidData && sessionDates[index][0] >= validStartDate) {
                return;
            }

            int[] items = sessionItems[index];
            long[] dates = sessionDates[index];
            int targetItem = sessionTargets[index];

            Set<Integer> uniqueItems = new HashSet<>();
            for (int item : items) {
                uniqueItems.add(item);
            }

            for (int item : uniqueItems) {
                double[] cache = this.itemCache[item];
                synchronized (cache) {
                    cache[SESSION_COUNT]++;
                    cache[AVG_SESS_LEN] += items.length;
                    cache[AVG_SESS_DUR] += RecSys22Helper.sessionDurationHR(dates);
                }
            }
            if (targetItem >= 0) {
                double[] cache = this.itemCache[targetItem];
                synchronized (cache) {
                    cache[BUY_COUNT]++;
                    cache[AVG_BOUGHT_SESS_LEN] += items.length;
                    cache[AVG_BOUGHT_SESS_DUR] += RecSys22Helper.sessionDurationHR(dates);
                }
            }

            long date = dates[0];
            for (int item1 : items) {
                long[] item1SessionDates = this.itemItemSessionDates[item1];
                for (int item2 : items) {
                    if (item1 == item2) {
                        continue;
                    }
                    synchronized (item1SessionDates) {
                        if (item1SessionDates[item2] < date) {
                            item1SessionDates[item2] = date;
                        }
                    }
                }
            }
        });
        IntStream.range(0, N_ITEMS).parallel().forEach(index -> {
            double[] cache = this.itemCache[index];
            if (cache[SESSION_COUNT] > 0) {
                cache[AVG_SESS_LEN] = cache[AVG_SESS_LEN] / cache[SESSION_COUNT];
                cache[AVG_SESS_DUR] = cache[AVG_SESS_DUR] / cache[SESSION_COUNT];
            }
            if (cache[BUY_COUNT] > 0) {
                cache[AVG_BOUGHT_SESS_LEN] = cache[AVG_BOUGHT_SESS_LEN] / cache[BUY_COUNT];
                cache[AVG_BOUGHT_SESS_DUR] = cache[AVG_BOUGHT_SESS_DUR] / cache[BUY_COUNT];
            }
        });

        TIMER.toc("initCache() done");
    }


    public float[] extractTargets(final RecSys22Session session) {
        float[] target = new float[session.candidateItems.length];
        boolean found = false;
        for (int i = 0; i < session.candidateItems.length; i++) {
            if (session.candidateItems[i] == session.targetItem) {
                target[i] = 1;
                found = true;
            }
        }
//        if (!found) {
//            throw new IllegalStateException("no positive target");
//        }
        return target;
    }

    public List<MLVector> extractItemFeatures(final int item,
                                              final boolean withContent) {
        List<MLVector> features = new ArrayList<>();
        double[] cache = this.itemCache[item];
        features.add(new MLVectorDenseFloat(new float[]{
                (float) cache[SESSION_COUNT],
                (float) cache[BUY_COUNT],
                (float) (cache[SESSION_COUNT] > 0 ? cache[BUY_COUNT] / cache[SESSION_COUNT] : 0f),
                (float) cache[AVG_SESS_LEN],
                (float) cache[AVG_SESS_DUR],
                (float) cache[AVG_BOUGHT_SESS_LEN],
                (float) cache[AVG_BOUGHT_SESS_DUR]
        }));

        if (withContent) {
            features.add(this.data.itemCatColSelect.getRow(item));
            features.add(this.data.itemCatValColSelect.getRow(item));
        }

        return features;
    }

    public List<MLVector> extractSessionFeatures(final RecSys22Session session) {
        List<MLVector> features = new ArrayList<>();
        int nItems = session.items.length;
        features.add(new MLVectorDenseFloat(new float[]{
                nItems,
                (float) RecSys22Helper.sessionDurationHR(session.dates),
        }));

        //content for last item
        int lastItem = session.items[session.items.length - 1];
        features.addAll(this.extractItemFeatures(lastItem, true));

        return features;
    }

    public List<MLVector> extractItemSessionFeatures(final RecSys22Session session,
                                                     final int targetItem) {
        List<MLVector> features = new ArrayList<>();

        int lastItem = session.items[session.items.length - 1];
        int secondLastItem = -1;
        if (session.items.length >= 2) {
            secondLastItem = session.items[session.items.length - 2];
        }
        double[] countsSession = new double[2];
        double[] countsLast = new double[2];

        int[] itemCats = this.data.itemCatIds[targetItem];
        int[] itemVals = this.data.itemCatVals[targetItem];
        for (int i = 0; i < session.items.length; i++) {
            int curItem = session.items[i];
            int[] curCats = this.data.itemCatIds[curItem];
            int[] curVals = this.data.itemCatVals[curItem];

            //overlap in content
            for (int j = 0; j < itemCats.length; j++) {
                for (int k = 0; k < curCats.length; k++) {
                    if (itemCats[j] != curCats[k]) {
                        continue;
                    }

                    //overlap in content categories
                    if ((j == 0 || itemCats[j] != itemCats[j - 1]) &&
                            (k == 0 || curCats[k] != curCats[k - 1])) {
                        countsSession[0]++;
                        if (i == (session.items.length - 1)) {
                            countsLast[0]++;
                        }
                    }

                    //overlap in content category-values
                    if (itemVals[j] == curVals[k]) {
                        countsSession[1]++;
                        if (i == (session.items.length - 1)) {
                            countsLast[1]++;
                        }
                    }
                }
            }
        }

        Set<Integer> uniqueItems = new HashSet<>();
        for (int item : session.items) {
            uniqueItems.add(item);
        }

        double[] buyCF = new double[2];
        double[] sessionCF = new double[2];
        for (int item : uniqueItems) {
            float score = RecSys22CF.getCFScore(targetItem, item, this.cf.itemItemBuy);
            buyCF[0] += score;
            if (buyCF[1] < score) {
                buyCF[1] = score;
            }
            score = RecSys22CF.getCFScore(targetItem, item, this.cf.itemItemSession);
            sessionCF[0] += score;
            if (sessionCF[1] < score) {
                sessionCF[1] = score;
            }
        }
        buyCF[0] = buyCF[0] / uniqueItems.size();
        sessionCF[0] = sessionCF[0] / uniqueItems.size();

        features.add(new MLVectorDenseFloat(new float[]{
                (float) (countsSession[0] / session.items.length),
                (float) (countsSession[1] / session.items.length),
                (float) countsLast[0],
                (float) countsLast[1],
                this.itemItemSessionDates[targetItem][lastItem] > 0 ?
                        (float) ((TRAIN_END_DATE - this.itemItemSessionDates[targetItem][lastItem]) / 86_400_000.0) : 0f,

                RecSys22CF.getCFScore(targetItem, lastItem, this.cf.itemItemBuy),
                RecSys22CF.getCFScore(targetItem, lastItem, this.cf.itemItemSession),

                secondLastItem >= 0 ? RecSys22CF.getCFScore(targetItem, secondLastItem,
                        this.cf.itemItemBuy) : 0f,
                secondLastItem >= 0 ? RecSys22CF.getCFScore(targetItem, secondLastItem,
                        this.cf.itemItemSession) : 0f,

                (float) buyCF[0],
                (float) buyCF[1],
                (float) sessionCF[0],
                (float) sessionCF[1]
        }));

        return features;
    }

    public List<MLVector>[] extractFeatures(final RecSys22Session session) {
        List<MLVector>[] features = new ArrayList[session.candidateItems.length];
        for (int i = 0; i < session.candidateItems.length; i++) {
            int candidateItem = session.candidateItems[i];
            features[i] = new ArrayList<>();

            //SESSION
            features[i].addAll(this.extractSessionFeatures(session));

            //ITEM
            features[i].addAll(this.extractItemFeatures(candidateItem, true));

            //SESSION-ITEM
            features[i].addAll(this.extractItemSessionFeatures(session, candidateItem));

        }
        return features;
    }

}
