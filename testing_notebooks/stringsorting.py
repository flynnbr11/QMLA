def sortmebaby(thestring):
    splitted = thestring.split("T")
    allsorted = [ "P".join(sorted(item.split("P")) ) for item in splitted    ]
    thestring = "T".join(allsorted)
    
    return thestring