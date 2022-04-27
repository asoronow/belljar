var ipc = require('electron').ipcRenderer;
var run = document.getElementById('run');
var indir = document.getElementById('indir');
var outdir = document.getElementById('outdir');

run.addEventListener('click', function(){
    ipc.once('maxResult', function(event, response){
        run.innerHTML = "Run";
    });

    ipc.once('maxError', function(event, response){
        run.innerHTML = "Run";
    });

    if (indir && outdir && indir.value && outdir.value) {
        run.classList.add('disabled');
        run.innerHTML = "<i class='fas fa-spinner fa-spin'></i>";
        ipc.send('runMax', [indir.value, outdir.value]);
    }
});
