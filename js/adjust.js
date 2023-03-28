var ipc = require('electron').ipcRenderer;
var run = document.getElementById('run');
var indir = document.getElementById('indir');
var loadbar = document.getElementById('loadbar');
var loadmessage = document.getElementById('loadmessage');
var back = document.getElementById('back');
var methods = document.getElementById('methods');
var affine = document.getElementById('affine');
var paint = document.getElementById('paint');
var selectedMethod = 'affine';


affine.addEventListener('click', function(){
    methods.textContent = 'Affine';
    selectedMethod = 'affine';
});

paint.addEventListener('click', function(){
    methods.textContent = 'Paint';
    selectedMethod = 'paint';
});

run.addEventListener('click', function(){
    if (indir && indir.value) {
        run.classList.add('disabled');
        back.classList.remove('btn-warning');
        back.classList.add('btn-danger')
        back.innerHTML = "Cancel";
        run.innerHTML = "<i class='fas fa-spinner fa-spin'></i>";
        ipc.send('runAdjust', [indir.value, selectedMethod]);
    }
});

back.addEventListener('click', function (event){
    if (back.classList.contains('btn-danger')){
        event.preventDefault();
        ipc.send('killAdjust', []);
        back.classList.add('btn-warning');
        back.classList.remove('btn-danger')
        back.innerHTML = "Back";
        run.innerHTML = "Run";
        run.classList.remove('disabled');
        loadmessage.innerHTML = "";
        loadbar.style.width = "0";
    }
});

ipc.on('adjustResult', function(event, response){
    run.innerHTML = "Run";
    run.classList.remove('disabled');
    back.classList.add('btn-warning');
    back.classList.remove('btn-danger')
    back.innerHTML = "Back";
    run.innerHTML = "Run";
    run.classList.remove('disabled');
    loadmessage.innerHTML = "";
    loadbar.style.width = "0";
});

ipc.once('adjustError', function(event, response){
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