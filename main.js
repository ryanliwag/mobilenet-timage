import * as tf from '@tensorflow/tfjs';

var webcamStream;

class Main {
  constructor() {

    // Initialize buttons
    this.open_cam = document.getElementById('open-webcam');
    this.open_cam.onclick = () => this.use_webcam();
    this.cap_cam = document.getElementById('capture-stream');
    this.cap_cam.onclick = () => this.snapshot();
    this.classify_img = document.getElementById('classify');
    this.classify_img.onclick = () => this.classify_model();

    $.getJSON("imagenet_class_index.json", function(json) {
      console.log(json[2][1]); // this will show the info it in firebug console
    });

 
    tf.loadModel('model/model.json').then((model) => {
      this.model = model;
      this.updateInputTensor();
    });
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
    canvas = document.getElementById("myCanvas");
    canvas.width="224";
    canvas.height="224";

    if (this.inputTensor) {
      this.inputTensor.dispose();
    }
    this.inputTensor = tf.tidy(() =>
      tf.fromPixels(canvas).toFloat().div(tf.scalar(255))
    );
    console.log(this.inputTensor.print)
    this.runmodel();
    


  }

  //call image net json file
  trigger_search(query) {
         var JSElement = document.createElement('script');
         JSElement.src = 'https://www.googleapis.com/customsearch/v1?key=AIzaSyCUwXHx7yvicNW3tAI-NAu47cKjN_4LFZ8&cx=001792966700025164735:8xnzg1hpiuc&q='+query+'&callback=hndlr';
         document.getElementsByTagName('body')[0].appendChild(JSElement);
    }

  runmodel() {
    const limit = tf.tensor1d([0.5]);
    console.log(this.inputTensor.expandDims())
    const prediction = this.model.predict(this.inputTensor.expandDims())
    var value = prediction.argMax(1).dataSync()
    console.log(prediction.reshape([1000]).dataSync()[value[0]])
    console.log(value[0])
  }


}

window.addEventListener('load', () => new Main());

