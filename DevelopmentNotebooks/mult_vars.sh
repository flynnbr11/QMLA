#!/bin/bash

further_growth_rules=(
	'growth_rule_1' 
	'growth_rule_2'
)

growth_rules=""
for item in ${further_growth_rules[*]}
do
    growth_rules+=" -l $item" 
done

echo $growth_rules


python3 test_parse.py $growth_rules
