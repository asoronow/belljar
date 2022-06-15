var ipc = require('electron').ipcRenderer;
var run = document.getElementById('run');
var preddir = document.getElementById('preddir');
var annodir = document.getElementById('annodir');
var outdir = document.getElementById('outdir');
var loadbar = document.getElementById('loadbar');
var loadmessage = document.getElementById('loadmessage');
var back = document.getElementById('back');

run.addEventListener('click', function(){
    if (preddir && annodir && outdir && preddir.value && annodir.value && outdir.value) {
        run.classList.add('disabled');
        back.classList.remove('btn-warning');
        back.classList.add('btn-danger')
        back.innerHTML = "Cancel";
        run.innerHTML = "<i class='fas fa-spinner fa-spin'></i>";
        ipc.send('runCount', [preddir.value, annodir.value, outdir.value]);
    }
});

back.addEventListener('click', function (event){
    if (back.classList.contains('btn-danger')){
        event.preventDefault();
        ipc.send('killCount', []);
        back.classList.add('btn-warning');
        back.classList.remove('btn-danger')
        back.innerHTML = "Back";
        run.innerHTML = "Run";
        run.classList.remove('disabled');
        loadmessage.innerHTML = "";
        loadbar.style.width = "0";
    }
});

ipc.once('countResult', function(event, response){
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

ipc.once('countError', function(event, response){
    run.innerHTML = "Run";
    run.classList.remove('disabled');
});

ipc.on('updateLoad', function (event, response) {
    loadbar.style.width = String(response[0]) + "%";
    loadmessage.innerHTML = response[1];
});

preddir.addEventListener('click', function(){
    ipc.once('returnPath', function(event, response){
        if (response[1] == 'preddir') {
            preddir.value = response[0];
        }
    })
    ipc.send('openDialog', 'preddir');
});

annodir.addEventListener('click', function(){
    ipc.once('returnPath', function(event, response){
        if (response[1] == 'annodir') {
            annodir.value = response[0];
        }
    })
    ipc.send('openDialog', 'annodir');
});

outdir.addEventListener('click', function(){
    ipc.once('returnPath', function(event, response){
        if (response[1] == 'outdir') {
            outdir.value = response[0];
        }
    })
    ipc.send('openDialog', 'outdir');
});