package inference;

import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

public class RecSys22ModelOptions {

    public Options options;

    public RecSys22ModelOptions() {
        this.options = new Options();

        Option option = new Option("action", true, "extractFeatures|trainModel|genSubmission");
        this.options.addOption(option);

        option = new Option("dataPath", true, "directory with serialized data");
        this.options.addOption(option);

        option = new Option("xgbModelPath", true, "directory for all XGB model files");
        this.options.addOption(option);

        option = new Option("xgbNCandidates", true, "number of item candidates to sample for XGB");
        this.options.addOption(option);

        option = new Option("xgbNTrees", true, "number of training rounds for XGB");
        this.options.addOption(option);

        option = new Option("xgbFirstStageModel", true, "first stage XGB model");
        this.options.addOption(option);

        option = new Option("xgbSecondStageModel", true, "second stage XGB model");
        this.options.addOption(option);

        option = new Option("xgbSecondStageTopN", true, "second stage top-N candidates for " +
                "training");
        this.options.addOption(option);

    }
}
