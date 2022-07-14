# Instructions

Final model scores are blended from 5 models from the 5 folder: `RecSys22`, `dae-2stage-inf`, `vae-cf-final`, `dot_product_final`, `sigmoid_final_refactor`.

1. Follow the `README.md` in each of the above 5 folders to produce the following 6 score files:
```
final_raw_scores.csv
sigmoid_final_score.csv
jianing_final_scores.csv
zhaolin_final_score_v1.csv
zhaolin_final_score_v2.csv
TestFinalSet_2000.model_1200_scores
```

2. Move the above scores files to the corresponding folder defined as `FINAL_FILES` in `RecSys22/src/main/java/blend/RecSys22Blender.java`. You can also modify it in code to adjust the path for your machine and move the score files to there.

3. `cd RecSys22` and run below command to blend the score files to have the final submission csv:
```
mvn clean compile
mvn exec:java -Dexec.mainClass="blend.RecSys22Blender" -Dexec.cleanupDaemonThreads=false -Dexec.classpathScope=compile
```
final submission file will then be saved under `/data/recsys2022/scores/blend/` by default but can be modified to adjust to your machine.

Same process for generating LB submission file under team name `LAYER 6`.
