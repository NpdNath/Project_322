var canvas, ctx;
var mouseX, mouseY, mouseDown = 0;
var touchX, touchY;

// function for interacting with canvas
function init() {
    canvas = document.getElementById('sketchpad');
    ctx = canvas.getContext('2d');
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (ctx) {
        canvas.addEventListener('mousedown', sketchpad_mouseDown, false);
        canvas.addEventListener('mousemove', sketchpad_mouseMove, false);
        window.addEventListener('mouseup', sketchpad_mouseUp, false);
        canvas.addEventListener('touchstart', sketchpad_touchStart, false);
        canvas.addEventListener('touchmove', sketchpad_touchMove, false);
    }
}

function draw(ctx, x, y, size, isDown) {
    if (isDown) {
        ctx.beginPath();
        ctx.strokeStyle = "purple";
        ctx.lineWidth = '10';
        ctx.lineJoin = ctx.lineCap = 'round';
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.closePath();
        ctx.stroke();
    }
    lastX = x;
    lastY = y;
}

// Event handlers

function sketchpad_mouseDown() {
    mouseDown = 1;
    draw(ctx, mouseX, mouseY, 12, false);
}

function sketchpad_mouseUp() {
    mouseDown = 0;
}

function sketchpad_mouseMove(e) {
    getMousePos(e);
    if (mouseDown == 1) {
        draw(ctx, mouseX, mouseY, 12, true);
    }
}

function getMousePos(e) {
    if (!e)
        var e = event;
    if (e.offsetX) {
        mouseX = e.offsetX;
        mouseY = e.offsetY;
    } else if (e.layerX) {
        mouseX = e.layerX;
        mouseY = e.layerY;
    }
}

function sketchpad_touchStart() {
    getTouchPos();
    draw(ctx, touchX, touchY, 12, false);
    event.preventDefault();
}

function sketchpad_touchMove(e) {
    getTouchPos(e);
    draw(ctx, touchX, touchY, 12, true);
    event.preventDefault();
}

function getTouchPos(e) {
    if (!e)
        var e = event;
    if (e.touches) {
        if (e.touches.length == 1) {
            var touch = e.touches[0];
            touchX = touch.pageX - touch.target.offsetLeft;
            touchY = touch.pageY - touch.target.offsetTop;
        }
    }
}

// Clearing the sketchpad

document.getElementById('clear_button').addEventListener("click", function () {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
});

// Integrating CANVAS with CNN MODEL

var base_url = window.location.origin;
let model;

(async function () {
    console.log("Model loading...");
    try {
      model = await tf.loadLayersModel("model.json");
      console.log("Model loaded successfully.");
    } catch (error) {
      console.error("Error loading model:", error);
    }
  })();

function preprocessCanvas(image) {
    let tensor = tf.browser.fromPixels(image)
      .resizeNearestNeighbor([28, 28])
      .mean(2)
      .expandDims(2)
      .expandDims()
      .toFloat();
    console.log("Preprocessed tensor shape:", tensor.shape);
    return tensor.div(255.0);
  }
  
// Prediction

document.getElementById('predict_button').addEventListener("click", async function () {
    var imageData = canvas.toDataURL();
    let tensor = preprocessCanvas(canvas);
    console.log("Preprocessed tensor:", tensor);
  
    try {
      let predictions = await model.predict(tensor).data();
      console.log("Prediction results:", predictions);
      displayLabel(results);
    } catch (error) {
      console.error("Error making prediction:", error);
    }
  });
  

// Output

function displayLabel(data) {
    var currencyLabels = ['Baht', 'Dollar', 'Euro', 'Yuan', 'Yen'];
    var max = data[0];
    var maxIndex = 0;
    for (var i = 1; i < data.length; i++) {
      if (data[i] > max) {
        maxIndex = i;
        max = data[i];
      }
    }
  
    var imgElement = document.createElement("img");
    imgElement.src = "images/" + currencyLabels[maxIndex] + ".png";
    imgElement.alt = "Prediction";
    imgElement.width = 100;
    imgElement.height = 100;
  
    var resultContainer = document.getElementById('result');
    resultContainer.innerHTML = '฿';
    resultContainer.appendChild(imgElement);
  }