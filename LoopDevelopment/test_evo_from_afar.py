import os as os
import sys as sys 

sys.path.append(os.path.join("..","Libraries","QML_lib"))
import Evo as evo


print("Evo dir : ")
print(dir(evo))

print(evo.identity())

