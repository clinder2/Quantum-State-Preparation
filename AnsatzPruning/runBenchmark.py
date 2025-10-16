import pandas as pd 

from AnsatzBenchmarking.Problems.maxCut.MaxCutProblems import MaxCutProblemSet
from AnsatzBenchmarking.Builders.fixedSU2 import FixedSU2Builder
from AnsatzBenchmarking.Builders.evolvedOp import EvolvedOperatorBuilder

from AnsatzBenchmarking.Builders.momentumBuilder import MomentumBuilder
from AnsatzBenchmarking.evaluator import evaluateBuilder

def main(): 
    problemSet = MaxCutProblemSet() 
    results1 = evaluateBuilder(MomentumBuilder, problemSet)
    df1 = pd.DataFrame(results1)
    print(df1)

    results2 = evaluateBuilder(FixedSU2Builder, problemSet)
    df2 = pd.DataFrame(results2)
    print(df2)


    results3 = evaluateBuilder(EvolvedOperatorBuilder, problemSet)
    df3 = pd.DataFrame(results3)
    print(df3)

    allResults = pd.concat([df1, df2, df3])
    allResults.to_csv("BenchmarkResults.csv")




if __name__ == "__main__": 
    main()