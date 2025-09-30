import pandas as pd 

from AnsatzBenchmarking.Problems.maxCut.MaxCutProblems import MaxCutProblemSet
from AnsatzBenchmarking.Builders.fixedSU2 import FixedSU2Builder
from AnsatzBenchmarking.evaluator import evaluateBuilder

def main(): 
    problemSet = MaxCutProblemSet() 
    results = evaluateBuilder(FixedSU2Builder, problemSet)
    df = pd.DataFrame(results)
    print(df)

if __name__ == "__main__": 
    main()
