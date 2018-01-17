import ipyparallel as ipp
c=ipp.Client()
print(c[:].apply_sync(lambda : "Hello, World"))
