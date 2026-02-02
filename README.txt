First activate venv
Then download requirements.txt
Select interpreter?

The project is organized by data, models, notebooks, and results.
Each model is implemented in a separate Python file to keep the comparison clear.

What could increase  model performance is more images, thats why as i said in the proposal, we need more data
how can I be sure to trust the author of the data
my images are color, meaning 3D tensors will be used, color layer (height, width and color) will be separated into 3 dimensions: result in height-width-red, height-width-green, height-width-blue
Images range between great differences: 137 x 121 to things like 46x 60 pixels - result in smaller images upsampled with larger images downsampled with using 128x128
keras is used: keras is a high-level library built on top of tensorflow, simple and intuitive

For future work, it would also be possible to use Transfer Learning, in order to leverage pre-trained transformers, for example for images.

pip install -r requirements.txt

For readme, since I didn't upload the dataset to github, use Jenny Yang's BeeImage dataset from Kaggle
https://www.kaggle.com/datasets/jenny18/honey-bee-annotated-images
is automatic download solvable with a script? Google Drive or Kaggle

https://www.youtube.com/watch?v=33ysE1Gt1G4&list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1&index=14

I could further proof it by using own images?

I could use a library that explains what the model considers important parts of the image

Explain in theis what the limitations are
Limitations of public datasets

Gap between academic and real-world ML

Ethical considerations

Future work for deployment


Start out with most basic CNN

Next up: 
- more cnn models
    - before that, fix classes (multiple, healthy, unhealthy)
- random forest, svm for images

For understanding I stopped at Cell 2

Deepak Meeting 02.02:
TODO
- Implement all three algorithms
- Present an initial comparison of the results
# More random for split in cnn
# Strategy to overcome overfitting
# Model loss
# Important: undestand output

Achieved:
- neater workspace organization
-- data, models, notebooks, and results
-- separate file for each model, so results are clean
- random forest
- further understanding

https://www.datacamp.com/tutorial/convolutional-neural-networks-python
https://www.datacamp.com/tutorial/pytorch-cnn-tutorial
https://www.datacamp.com/tutorial/random-forests-classifier-python
https://www.kaggle.com/code/prashant111/random-forest-classifier-tutorial

TODO:
- clean cnn of exploration and data cleaning
- svm
- halve the healthy or make unhealthy bigger -> balance the classes
- after conclusion, try to come up with a concrete result and a reason, why one model performs better or worse, focus on main conclusion, RQ
read and think of strengths, weaknesses in literature, see if it applies
Feb 12th Feb 15:00