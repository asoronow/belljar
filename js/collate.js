var ipc = require('electron').ipcRenderer;
var run = document.getElementById('run');
var indir = document.getElementById('indir');
var outdir = document.getElementById('outdir');
var loadbar = document.getElementById('loadbar');
var loadmessage = document.getElementById('loadmessage');
var back = document.getElementById('back');
var regions = document.getElementById('regions');

function checkNumber(value, message) {
    var str = value.toString();
    if(!str.match(/^-?\d*\.?\d*$/)) {
        alert(`${message}`);
        return false;
    }
    return true;
}

run.addEventListener('click', function(){
    if (indir && outdir && indir.value && outdir.value) {
        run.classList.add('disabled');
        back.classList.remove('btn-warning');
        back.classList.add('btn-danger')
        back.innerHTML = "Cancel";
        run.innerHTML = "<i class='fas fa-spinner fa-spin'></i>";
        ipc.send('runCollate', [indir.value, outdir.value, regions.innerText]);
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

ipc.on('collateResult', function(event, response){
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
    ipc.send('openFileDialog', 'indir');
});

outdir.addEventListener('click', function(){
    ipc.once('returnPath', function(event, response){
        if (response[1] == 'outdir') {
            outdir.value = response[0];
        }
    })
    ipc.send('openDialog', 'outdir');
});