package inference;

import org.slf4j.LoggerFactory;
import split.MLSplit;
import utils.MLTimer;

import java.util.HashMap;
import java.util.Map;

public class RecSys22Split extends MLSplit {

    public static MLTimer TIMER;

    static {
        MLTimer.initDefaultLogger();
        TIMER = new MLTimer(LoggerFactory.getLogger(RecSys22Split.class));
    }

    private static final long serialVersionUID = 6087477885808305148L;
    public static String TRAIN_SET = MLSplit.TRAIN_SET;
    public static String VALID_SET = MLSplit.VALID_SET;
    public static String TEST_LB_SET = "TestLBSet";
    public static String TEST_FINAL_SET = "TestFinalSet";

    public static final String TRAIN_START_DATE = "2020-01-01 00:00:00";
    public static final String TRAIN_END_DATE = "2021-06-01 00:00:00";

    public Map<String, int[]> splitSessionIds;
    public Map<String, int[][]> splitSessionItems;
    public Map<String, long[][]> splitSessionDates;
    public Map<String, int[]> splitSessionTargets;
    public Map<String, int[]> splitCandidates;

    public RecSys22Split() {
        this.splitSessionIds = new HashMap<>();
        this.splitSessionItems = new HashMap<>();
        this.splitSessionDates = new HashMap<>();
        this.splitSessionTargets = new HashMap<>();
        this.splitCandidates = new HashMap<>();
    }
}
