# ML_NLP

## Instructions

### Part 2 

Calculate the output for the datasets. 

To run: 

    python source/part2/part2.py [dataset]/[training file] [dataset]/[testing file] [dataset]/[output file]

or as an example: 

    python source/part2/part2.py EN/train EN/dev.in EN/dev.p2.out

The output will be in the output folder. Compare the answers with the evaluation script.
    
    
    
### Evaluation Script

To evaluate, run:

    python evalResult.py [dataset]/dev.out [dataset]/[output file]
    
or

    python evalResult.py EN/dev.out EN/dev.p2.out