package inference;

import java.time.ZonedDateTime;

public class RecSys22ModelConfig {
    public String dataPath;

    public String xgbModelPath;
    public String xgbFirstStageModel;

    public int xgbNCandidates;
    public int xgbNTrees;

    public ZonedDateTime validStartDateTime;
    public boolean removeValidData;
}
