# Sign-Language-Identification
A project of CS601.475 Machine Learning (JHU) that aims to recognize alphabetic letters in American Sign Language.

![1BN0$2_{U EPMMPWEX~KAWH](https://user-images.githubusercontent.com/77927150/167261493-a3a6e36a-b8ae-4f97-ba28-2d54457765d8.png)

# How to run hand_localizer.py

modify the main function:
```
  # load the binary classifier for hand detection using this line
  handDetector = loadClassifier('./handDetector.pkl')
  # load the nulticlass classifier for ASL recigniztion using this line
  signDetector = loadClassifier('./signDetector.pkl')

  # load the model using this line, the input argument is the directory storing the datasets
  gs = GestureRecognizer('/content/drive/MyDrive/ML_final/Sliding_window/')
  userlist=[ 'user_3',
        'user_4','user_5','user_6','user_7','user_9','user_10']

  # set the training data and testing data
  user_tr = userlist[:1]
  user_te = userlist[-1:]
  
  # train the model
  gs.train(user_tr)
  
  # save trained model
  gs.save_model(name = "sign_detector.pkl.gz", version = "0.0.1", author = 'Gill')
  
  # load saved model
  new_gr = GestureRecognizer.load_model(name = "/content/drive/MyDrive/Sliding_window/sign_detector.pkl.gz") # automatic dict unpacking
  print (new_gr.label_encoder)
  print (new_gr.signDetector) 

  # use model to test on new test data
  data = glob.glob('/content/drive/MyDrive/Dataset/user_10'+'/*.jpg')
  for i in range(240):
    z = io.imread(data[i])
    z = np.array(z)
    print (z.shape) 
    new_gr.recognize_gesture(np.array(z))
```

then run the file using 
```
!python hand_localizer.py
```
