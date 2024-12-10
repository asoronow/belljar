var ipc = require('electron').ipcRenderer;

var log = document.getElementById('log');


function makeSpan(text){
    // Make lines separeted by <br> and add span
    return `<br></br><span>${text}</span>`;
}

function cacheLog(){
    // store log between refreshes
    var log = document.getElementById('log');
    var logText = log.innerHTML;
    localStorage.setItem('log', logText);
}

function loadLog(){
    // load log from cache
    var log = document.getElementById('log');
    var logText = localStorage.getItem('log');
    log.innerHTML = logText;
}

function clearCache(){
    // clear log cache
    localStorage.removeItem('log');
}

function checkLogExpiry(){
    var expiry = new Date();
    expiry.setDate(expiry.getDate() - 1);
    var expiryTime = expiry.getTime();
    var logTime = localStorage.getItem('logTime');
    if (logTime == null){
        cacheLogTime();
        clearCache();
        loadLog();
    }

    if(logTime < expiryTime){
        clearCache();
    } else {
        loadLog();
    }
}

function cacheLogTime(){
    // store log time
    var logTime = new Date().getTime();
    localStorage.setItem('logTime', logTime);
}

// on load
window.onload = function(){
    checkLogExpiry();
}

ipc.on('savelogs', function(event, response){
    cacheLog();
    cacheLogTime();
});

ipc.on('log', function(event, response){
    console.log(response);
    // Insert log message as span in log
    // Split response text by newlines
    let lines = response.split('\n');
    // Make each line a span
    lines.forEach(function(line){
        log.innerHTML += makeSpan(line);
    });
    // Scroll to bottom of log
    window.scrollTo(0,document.body.scrollHeight);
});