if (!VelocityLimitPublisher) {
    var VelocityLimitPublisher = {
        ros: null,
        name: "",
        init: function() {
            this.ros = new ROSLIB.Ros();
            this.ros.on('error', function(error) {
                document.getElementById('velocity_limit_info').innerHTML = "Error";
            });
            this.ros.on('connection', function(error) {
                document.getElementById('velocity_limit_info').innerHTML = "Connected";
            });
            this.ros.on('close', function(error) {
                document.getElementById('velocity_limit_info').innerHTML = "Closed";
            });
            this.ros.connect('ws://' + location.hostname + ':9090');
        },
        send: function() {
            var pub = new ROSLIB.Topic({
                ros: this.ros,
                name: '/planning/scenario_planning/motion_velocity_optimizer/external_velocity_limit_mps',
                messageType: 'std_msgs/Float32'
            });
            var str = new ROSLIB.Message({
                data: parseFloat(velocity_limit_form.velocity_limit.value) / 3.6
            });
            pub.publish(str);
        }
    }
    VelocityLimitPublisher.init();

    window.onload = function() {};
    window.onunload = function() {
        VelocityLimitPublisher.ros.close();
    };
}
if (!VelocityLimitSubscriber) {
    var VelocityLimitSubscriber = {
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
                name: '/planning/scenario_planning/motion_velocity_optimizer/external_velocity_limit_mps',
                messageType: 'std_msgs/Float32'
            });
            sub.subscribe(function(message) {
                const div = document.getElementById("velocity_limit_status");
                if (div.hasChildNodes()) {
                    div.removeChild(div.firstChild);
                }
                var res = message.data;
                var el = document.createElement("span");
                el.innerHTML = res * 3.6;
                document.getElementById("velocity_limit_status").appendChild(el);
            });
        }
    }
    VelocityLimitSubscriber.init();

    window.onload = function() {};
    window.onunload = function() {
        VelocityLimitSubscriber.ros.close();
    };
}