# ML_NLP

## Instructions

### Part 2 

Calculate the output for the datasets. 

To run: 

    python source/part2/part2.py datasets/[dataset]/[training file] datasets/[dataset]/[testing file] output/[output file]

or as an example: 

    python source/part2/part2.py datasets/EN/train datasets/EN/dev.in output/dev.p2.out

The output will be in the output folder. Compare the answers with the evaluation script.
    
    
    
### Evaluation Script

To evaluate, run:

    python evalResult.py datasets/[dataset]/dev.out output/[output file]
    
or

    python evalResult.py datasets/EN/dev.out output/dev.p2.out
