import * as tf from '@tensorflow/tfjs';

var webcamStream;
class Main {
    constructor() {
    // Initialize buttons
    this.initialize_image();

    this.open_cam = document.getElementById('open-webcam');
    this.open_cam.onclick = () => this.use_webcam();
    this.cap_cam = document.getElementById('capture-stream');
    this.cap_cam.onclick = () => this.snapshot();
    this.classify_img = document.getElementById('classify');
    this.classify_img.onclick = () => this.classify_model();

    tf.loadModel('model/model.json').then((model) => {
      this.model = model;
    });
  }

  initialize_image() {
    var canvas = document.getElementById('myCanvas');
    var context = canvas.getContext('2d');
    var imageObj = new Image();
    imageObj.onload = function() {
      context.drawImage(imageObj,0, 0);
    };
    imageObj.src = 'goldfish.jpeg';
  }

  use_webcam() {
    this.add_video()
    var video = document.getElementById('video')
    if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true, audio: false }).then(function(localMediaStream) { 
      video.src = window.URL.createObjectURL(localMediaStream);
      webcamStream = localMediaStream;
      });
     }
  }
  
  snapshot() {
    var canvas, ctx;
    canvas = document.getElementById("myCanvas");
    ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0,0, canvas.width, canvas.height);
    console.log(canvas.getContext('2d').getImageData(0, 0, 400, 400).data);

    this.close_webcam();
  }

  close_webcam() {
    var elem = document.getElementById('video');
    webcamStream.getVideoTracks()[0].stop();
    elem.parentNode.removeChild(elem);
  }

  add_video() {
    var video = document.createElement('video');
    video.autoplay = true;
    video.id = "video";
    video.width = "640";
    video.height = "480";
    document.getElementById('video_loader').appendChild(video);
  }

  classify_model(canvas) {
    var canvas = document.getElementById("myCanvas");
    var ctx = canvas.getContext('2d');
    var imgData=ctx.getImageData(0,0,224,224);
    if (this.inputTensor) {
      this.inputTensor.dispose();
    }
    this.inputTensor = tf.tidy(() =>
      tf.fromPixels(imgData).toFloat().div(tf.scalar(255))
    );
    console.log(this.inputTensor.print)
    this.runmodel();
  }

  runmodel() {
    const limit = tf.tensor1d([0.5]);
    console.log(this.inputTensor.expandDims())
    const prediction = this.model.predict(this.inputTensor.expandDims())
    var value = prediction.argMax(1).dataSync()
    console.log(prediction.reshape([1000]).dataSync()[value[0]])
    console.log(value[0])

    $.getJSON("imagenet_class_index.json", function(json) {
      document.getElementById('result').innerHTML = json[value[0]][1]; 
    });
  }
}

window.addEventListener('load', () => new Main());

