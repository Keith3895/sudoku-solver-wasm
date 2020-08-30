const cv = require('./lib/opencv.js');
console.log(cv);
const videoElement = document.getElementById('webcam');
const videoTargetCanvas = document.getElementById('videoCanvas');
const videoTargetCanvasOut = document.getElementById('videoCanvasOUT');
let src;
let dst;
let cap;
const startCamera = async () => {
    await navigator.mediaDevices.getUserMedia({
        // audio: false,
        video: true
    }).then(stream => {
        const videoSettings = stream.getVideoTracks()[0].getSettings();
        videoTargetCanvas.width = videoSettings.width;
        videoTargetCanvas.height = videoSettings.height;
        videoElement.srcObject = stream;
        videoElement.play();
        

        // debugger
    }).catch(function (err) {
        console.log("An error occured! " + err);
    });
}
cv["onRuntimeInitialized"] = async () => {
    await startCamera();
    alert('hello world', videoElement);
    // while (true) {
    window.requestAnimationFrame(runCanvas);
    // }

};

function runCanvas() {
    let context = videoTargetCanvas.getContext("2d");
    context.drawImage(videoElement, 0, 0);
    src = new cv.Mat(videoElement.height, videoElement.width, cv.CV_8UC4);
    dst = new cv.Mat(videoElement.height, videoElement.width, cv.CV_8UC1);
    src.data.set(context.getImageData(0, 0, videoElement.width, videoElement.height).data);
    cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
    cv.imshow("videoCanvasOUT", dst); 
    // cap = new cv.VideoCapture(videoElement);
    // const tmp = new cv.Mat(dst);
    // if (tmp.type() === cv.CV_8UC1) {
    //     cv.cvtColor(tmp, tmp, cv.COLOR_GRAY2RGBA);
    // } else if (tmp.type() === cv.CV_8UC3) {
    //     cv.cvtColor(tmp, tmp, cv.COLOR_RGB2RGBA);
    // }
    // const imgData = new ImageData(
    //     new Uint8ClampedArray(dst.data),
    //     dst.cols,
    //     dst.rows
    // );
    // videoTargetCanvas.getContext("2d").putImageData(imgData, 0, 0);
    // videoTargetCanvas.getContext("2d").drawImage(videoElement, 0, 0);
    // console.log(imgRead(videoTargetCanvas));
    window.requestAnimationFrame(runCanvas);
}