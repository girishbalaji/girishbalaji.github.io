# Repository Structure
The repository is structured very simply: all the input data (including that I collected from online) for the bonus images are in the ./data folder; the rest is in ./ main repository. As the images are the main deliverable, the output images are also stored in the main repository along with the code.

The `proj1.ipynb` is a Python notebook that you can run through sequentially to see how the entire code works.

The `main.py` is as that requested; it simply took all the code in the ipython notebook and runs it sequentially. Note that when this file is run it will output a plt.show() a bunch of images. The user will need to close the screen with the image displayed in order to keep the script running.

IMPORTANT: Note running either of the files will not work as you do not have the input data. The data has specific names which are used in both the ipynb and the main.py. Please email me if you want the data folder. The data folder also contains data that I chose off the Library of Congress.

If you do have the data all named correctly, you would be able to run the cells in the ipynb or the main.py file and it will process all the images.

The bells and whistles implemented are in both files. Note there is not explicitly separate code for the bells and whistles. Though there is an explicit "if condition" i.e. if `name == Emir` or the `hist_eq = True` or the flag is set, it will run that particular cool function.

Also, the `align_img` function takes in a file, filename.jpg in the ./data folder and returns out_filename.jpg in the . folder. So if you run `align_img` on the same picture, it will replace the output image.
