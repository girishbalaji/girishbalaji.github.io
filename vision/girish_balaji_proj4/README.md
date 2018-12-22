# repository structure

### output/ - final image outputs s
### data/ - input data
### dane_data/ - images from danish dataset
### dane_not_smiling/ - copy of images from dane_data/ not smiling
### dane_smiling/ - copy of images from dane_data/ smiling

How to use main.py?
Have one source image, target image to morph.
In the document, call IM1_NAME, IM2_NAME, the name of those two respectively.
Edit the prepreocessing code if need be to match the size.
Set the part of the project you want to run to True and set the rest to False.


You need to work with the header on top of main.py and adjust those before anytime you run python main.py.

```
PREPROCESS = False

# data/edited_{im1_name} and data/edited_{im2_name} need to exist
GEN_REF = False
GEN_IM2_PTS = False

# Points Exist Needs to be True to run GET_AVG and GET_FULL_MORPH
POINTS_EXIST = False
GET_AVG = False
GET_FULL_MORPH = False

DANE_DATA = False

# Adds Caricature messup to IM2
GET_CARICATURE = True

# Adds smile to IM2
MORE_SMILING = False

# default: me_glass, satya
IM1_NAME = "avg_dane"
IM2_NAME = "me_no_glasses_for_dane"
```


If you set the IM1_NAME and IM2_NAME correctly and set up the same directory structure, and set the correct section to True (for all new images make sure to go in order), it should execute correctly.


For the missing directories please contact me. I couldn't turn in everything because of space requirements.

Download the danish data from online and put it in the submission repository as `dane_data`
