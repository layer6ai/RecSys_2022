#!/bin/bash
export MAVEN_OPTS="-Xmx200g -Xms200g"

mvn clean compile
mvn exec:java -Dexec.mainClass="inference.RecSys22Data" -Dexec.args="/data/recsys2022/data/" -Dexec.cleanupDaemonThreads=false -Dexec.classpathScope=compile
mvn exec:java -Dexec.mainClass="inference.RecSys22Model" -Dexec.args="-action extractFeatures -dataPath /data/recsys2022/data/ -xgbModelPath /data/recsys2022/model/xgb/ -xgbNCandidates 20" -Dexec.cleanupDaemonThreads=false -Dexec.classpathScope=compile
mvn exec:java -Dexec.mainClass="inference.RecSys22Model" -Dexec.args="-action trainModel -dataPath /data/recsys2022/data/ -xgbModelPath /data/recsys2022/model/xgb/ -xgbNTrees 2000 -xgbFirstStageModel 2000.model" -Dexec.cleanupDaemonThreads=false -Dexec.classpathScope=compile
mvn exec:java -Dexec.mainClass="inference.RecSys22Model" -Dexec.args="-action genSubmission -dataPath /data/recsys2022/data/ -xgbModelPath /data/recsys2022/model/xgb/ -xgbNTrees 1200 -xgbFirstStageModel 2000.model" -Dexec.cleanupDaemonThreads=false -Dexec.classpathScope=compile