let this_thing=10
a=1

if [ $a == 1 ]
then
    echo "Changing value of $this_thing"
    let this_thing="$this_thing/5"
    
    echo "Now this thing: $this_thing"
fi
