import os  
import xml.etree.ElementTree as ET
docset=[]
for fn in os.listdir('.'):
     if os.path.isfile(fn):
        if fn.endswith(".xml"):
        	tree = ET.parse(fn)
		root = tree.getroot()
for child in root:
	print child[2].text
	docset.append(child[2].text)		
