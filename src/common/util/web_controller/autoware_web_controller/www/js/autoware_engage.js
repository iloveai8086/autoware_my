if (!AutowareEngagePublisher) {
    var AutowareEngagePublisher = {
        ros: null,
        name: "",
        init: function() {
            this.ros = new ROSLIB.Ros();
            this.ros.on('error', function(error) {
                document.getElementById('autoware_engage_info').innerHTML = "Error";
            });
            this.ros.on('connection', function(error) {
                document.getElementById('autoware_engage_info').innerHTML = "Connected";
            });
            this.ros.on('close', function(error) {
                document.getElementById('autoware_engage_info').innerHTML = "Closed";
            });
            this.ros.connect('ws://' + location.hostname + ':9090');
        },
        send: function() {
            var pub = new ROSLIB.Topic({
                ros: this.ros,
                name: '/autoware/engage',
                messageType: 'std_msgs/Bool',
                latch: 'true'
            });

            var str = new ROSLIB.Message({
                data: true
            });
            pub.publish(str);
        }
    }
    AutowareEngagePublisher.init();

    window.onload = function() {};
    window.onunload = function() {
        AutowareEngagePublisher.ros.close();
    };
}
if (!AutowareDisengagePublisher) {
    var AutowareDisengagePublisher = {
        ros: null,
        name: "",
        init: function() {
            this.ros = new ROSLIB.Ros();
            this.ros.on('error', function(error) {
                document.getElementById('state').innerHTML = "Error";
            });
            this.ros.on('connection', function(error) {
                document.getElementById('state').innerHTML = "Connected";
            });
            this.ros.on('close', function(error) {
                document.getElementById('state').innerHTML = "Closed";
            });
            this.ros.connect('ws://' + location.hostname + ':9090');
        },
        send: function() {
            var pub = new ROSLIB.Topic({
                ros: this.ros,
                name: '/autoware/engage',
                messageType: 'std_msgs/Bool',
                latch: 'true'
            });

            var str = new ROSLIB.Message({
                data: false
            });
            pub.publish(str);
        }
    }
    AutowareDisengagePublisher.init();

    window.onload = function() {};
    window.onunload = function() {
        AutowareDisengagePublisher.ros.close();
    };
}
if (!AutowareEngageStatusSubscriber) {
    var AutowareEngageStatusSubscriber = {
        ros: null,
        name: "",
        init: function() {
            this.ros = new ROSLIB.Ros();
            this.ros.on('error', function(error) {
                document.getElementById('state').innerHTML = "Error";
            });
            this.ros.on('connection', function(error) {
                document.getElementById('state').innerHTML = "Connect";
            });
            this.ros.on('close', function(error) {
                document.getElementById('state').innerHTML = "Close";
            });
            this.ros.connect('ws://' + location.hostname + ':9090');

            var sub = new ROSLIB.Topic({
                ros: this.ros,
                name: '/autoware/engage',
                messageType: 'std_msgs/Bool'
            });
            sub.subscribe(function(message) {
                const div = document.getElementById("autoware_engage_status");
                if (div.hasChildNodes()) {
                    div.removeChild(div.firstChild);
                }
                var res = message.data;
                var el = document.createElement("span");
                el.innerHTML = res
                document.getElementById("autoware_engage_status").appendChild(el);
            });
        }
    }
    AutowareEngageStatusSubscriber.init();

    window.onload = function() {};
    window.onunload = function() {
        AutowareEngageStatusSubscriber.ros.close();
    };
}
