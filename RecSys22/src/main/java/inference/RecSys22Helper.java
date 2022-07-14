package inference;

import utils.MLRandomUtils;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.time.temporal.ChronoField;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class RecSys22Helper {

    public static final DateTimeFormatter DATE_FORMATTER = new DateTimeFormatterBuilder()
            .appendPattern("yyyy-MM-dd HH:mm:ss")
            .appendFraction(ChronoField.MILLI_OF_SECOND, 0, 3, true)
            .toFormatter();
    public static final ZoneId GMT = ZoneId.of("GMT");

    public static long convertToMillis(final String date) {
        return convertToDateTime(date).toInstant().toEpochMilli();
    }

    public static ZonedDateTime convertToDateTime(final String date) {
        return LocalDateTime.parse(date, RecSys22Helper.DATE_FORMATTER).atZone(RecSys22Helper.GMT);
    }

    public static boolean isInInterval(final long[] sessionDates,
                                       final long intervalStartDate,
                                       final long intervalEndDate) {
        return (sessionDates[0] >= intervalStartDate && sessionDates[0] < intervalEndDate);
    }

    public static double sessionDurationHR(final long[] sessionDates) {
        if (sessionDates.length <= 1) {
            return 0;
        }
        long diff = sessionDates[sessionDates.length - 1] - sessionDates[0];
        if (diff < 0) {
            throw new IllegalStateException("negative duration");
        }
        return diff / 3600000.0;
    }

    public static int[] getAllCandidates(final long[][] sessionDates,
                                         final int[] sessionTargets,
                                         final long startDate,
                                         final long endDate) {
        Set<Integer> candidates = new HashSet<>();
        for (int i = 0; i < sessionDates.length; i++) {
            if (RecSys22Helper.isInInterval(sessionDates[i], startDate, endDate)) {
                candidates.add(sessionTargets[i]);
            }
        }
        int[] candidatesArr = candidates.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(candidatesArr);
        return candidatesArr;
    }

    public static int[] sampleCandidates(final int targetItem,
                                         final int[] sessionItems,
                                         final int[] candidates,
                                         final int seed,
                                         final int nCandidates) {
        int[] cloneCandidates = candidates.clone();
        MLRandomUtils.shuffle(cloneCandidates, seed);

        int[] cloneSession = sessionItems.clone();
        Arrays.sort(cloneSession);

        int cur = 0;
        int[] sample = new int[nCandidates + 1];
        boolean inSample = false;
        for (int sampleItem : cloneCandidates) {
            //exclude session items from sample
            if (Arrays.binarySearch(cloneSession, sampleItem) < 0) {
                sample[cur] = sampleItem;
                if (sampleItem == targetItem) {
                    //check if target item is in the sample
                    inSample = true;
                }
                cur++;
                if (cur == (nCandidates + 1)) {
                    break;
                }
            }
        }
        if (cur != (nCandidates + 1)) {
            throw new IllegalStateException("not enough sample");
        }
        if (!inSample) {
            //not in sample so add it
            sample[0] = targetItem;
        }
        Arrays.sort(sample);

        return sample;
    }

    public static int[] removeSessionItems(final int[] sessionItems,
                                           final int[] candidates) {
        int[] cloneSession = sessionItems.clone();
        Arrays.sort(cloneSession);

        int cur = 0;
        int[] newCandidates = new int[candidates.length];
        for (int candidate : candidates) {
            if (Arrays.binarySearch(cloneSession, candidate) < 0) {
                newCandidates[cur] = candidate;
                cur++;
            }
        }
        if (cur < candidates.length) {
            newCandidates = Arrays.copyOfRange(newCandidates, 0, cur);
        }
        return newCandidates;
    }

    public static void sampleSession(final RecSys22Session session,
                                     final int seed) {
        //see "Constructing Data for Test Sessions" here
        //http://www.recsyschallenge.com/2022/dataset.html
        if (session.items.length < 2) {
            return;
        }

        final float MIN = 0.5f;
        final float MAX = 1.0f;

        float sample = new Random(seed).nextFloat();
        sample = MIN + (MAX - MIN) * sample;

        int newLength = Math.round(session.items.length * sample);
        session.items = Arrays.copyOfRange(session.items, 0, newLength);
        session.dates = Arrays.copyOfRange(session.dates, 0, newLength);
    }
}
