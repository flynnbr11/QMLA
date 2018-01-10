import numpy as np
import itertools
import Evo as evo
import warnings


def FloorPruningRule(modeltest, floor_thresh = 0.1): #len(self.ModelsList)

    for i in range(len(modeltest.ModelsList)):
        if(modeltest.ModelsList[i].KLogTotLikelihood is []):
            warn("\nFloorPruningRule called before a list 'KLogTotLikelihood' was made available", UserWarning)
    
    array_KLogTotLikelihood = np.array(list(map(lambda model: model.KLogTotLikelihood, modeltest.ModelsList)))
    renorm_KLogTotLikelihood = abs(1/array_KLogTotLikelihood)/abs(sum(1/abs(array_KLogTotLikelihood) ))

    del_list = []
    
    for i in range(len(modeltest.ModelsList)):
        if renorm_KLogTotLikelihood[i] < floor_thresh:
            modeltest.DiscModelsList.append(modeltest.ModelsList[i])
            modeltest.DiscModsOpList.append(modeltest.ModelsList[i])
            modeltest.DiscModelNames.append(modeltest.ModelNames[i])
            modeltest.DiscModelDict.update({ modeltest.ModelNames[i]: modeltest.ModelDict[modeltest.ModelNames[i]] })
            
            del_list.append(i)
    
       
    del_list.sort(reverse=True)
    for index in del_list:
        print('Model ' + str(modeltest.ModelNames[index]) + ' discarded upon FloorPruningRule')
        
        del(modeltest.ModelsList[index])
        del(modeltest.ModsOpList[index])
        del(modeltest.ModelDict[modeltest.ModelNames[index] ])
        del(modeltest.ModelNames[index])
            
            
            
            
            
            
def DetectSaturation(modeltest, use_datalength = 10, saturate_STN=3.):            
    
    detect_events = [False]*len(modeltest.ModelsList)
    
    for i in range( len(modeltest.ModelsList) ):
    
        volume_list = modeltest.ModelsList[i].VolumeList
        volume_data = volume_list[0:use_datalength]
                
        slope = np.polyfit(range(len(volume_data)), volume_data, deg=1)[0]
        volume_data_flatten = np.array([ list(map(lambda i: volume_data[i]  - slope*i, range( len(volume_data) ) )) ]) 
        noise = np.std(volume_data_flatten)
        
        if abs(noise)>abs(saturate_STN*slope):
            detect_events[i] = True
            
    return(detect_events)
        
        