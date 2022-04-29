var ipc = require('electron').ipcRenderer;
var run = document.getElementById('run');
var indir = document.getElementById('indir');
var outdir = document.getElementById('outdir');
var loadbar = document.getElementById('loadbar');
var tile = document.getElementById('tile');
var confidence = document.getElementById('confidence');
var model = document.getElementById('model');
var loadmessage = document.getElementById('loadmessage');
var back = document.getElementById('back');
var advance = document.getElementById('advance');
var arrow = document.getElementById('arrow');

advance.addEventListener('click', function () {
    arrow.classList.toggle('down');
});

function checkNumber(value, message) {
    var str = value.toString();
    if(!str.match(/^-?\d*\.?\d*$/)) {
        alert(`${message}`);
        return false;
    }
    return true;
}

run.addEventListener('click', function(){
    var c = 0.5;
    var t = 640;
    var m = '';

    if (indir && outdir && indir.value && outdir.value) {
        if (confidence.value && confidence.value < 1 && confidence.value > 0) {
            c = (checkNumber(gamma, "Confidence should be a float between 0-1, using default.")) ? confidence.value:0.5
        }
        if (tile.value && tile.value > 0) {
            t = (checkNumber(gamma, "Tile should be an integer, using default.")) ? tile.value:640
        }
        if (model.value) {
            m = model.value
        }
        run.classList.add('disabled');
        back.classList.remove('btn-warning');
        back.classList.add('btn-danger')
        back.innerHTML = "Cancel";
        run.innerHTML = "<i class='fas fa-spinner fa-spin'></i>";
        ipc.send('runDetection', [indir.value, outdir.value, c, t, m]);
        loadmessage.innerHTML = "Intializing...";
    }
});

back.addEventListener('click', function (event){
    if (back.classList.contains('btn-danger')){
        event.preventDefault();
        ipc.send('killDetect', []);
        back.classList.add('btn-warning');
        back.classList.remove('btn-danger')
        back.innerHTML = "Back";
        run.innerHTML = "Run";
        run.classList.remove('disabled');
        loadmessage.innerHTML = "";
        loadbar.style.width = "0";
    }
});

ipc.once('detectResult', function(event, response){
    back.classList.add('btn-warning');
    back.classList.remove('btn-danger')
    back.innerHTML = "Back";
    run.innerHTML = "Run";
    run.classList.remove('disabled');
    loadmessage.innerHTML = "";
    loadbar.style.width = "0";
});

ipc.once('detectError', function(event, response){
    run.innerHTML = "Run";
    run.classList.remove('disabled');
});

ipc.on('updateLoad', function (event, response) {
    loadbar.style.width = String(response[0]) + "%";
    loadmessage.innerHTML = response[1];
});

indir.addEventListener('click', function(){
    ipc.once('returnPath', function(event, response){
        if (response[1] == 'indir') {
            indir.value = response[0];
        }
    })
    ipc.send('openDialog', 'indir');
});

outdir.addEventListener('click', function(){
    ipc.once('returnPath', function(event, response){
        if (response[1] == 'outdir') {
            outdir.value = response[0];
        }
    })
    ipc.send('openDialog', 'outdir');
});

model.addEventListener('click', function(){
    ipc.once('returnPath', function(event, response){
        if (response[1] == 'model') {
            model.value = response[0];
        }
    })
    ipc.send('openFileDialog', 'model');
});