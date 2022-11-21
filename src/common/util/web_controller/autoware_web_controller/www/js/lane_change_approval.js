if (!LaneChangeApprovalPublisher) {
    var LaneChangeApprovalPublisher = {
        ros: null,
        name: "",
        init: function() {
            this.ros = new ROSLIB.Ros();
            this.ros.on('error', function(error) {
                document.getElementById('lane_change_approval_info').innerHTML = "Error";
            });
            this.ros.on('connection', function(error) {
                document.getElementById('lane_change_approval_info').innerHTML = "Connected";
            });
            this.ros.on('close', function(error) {
                document.getElementById('lane_change_approval_info').innerHTML = "Closed";
            });
            this.ros.connect('ws://' + location.hostname + ':9090');
        },
        send: function() {
            var pub = new ROSLIB.Topic({
                ros: this.ros,
                name: '/lane_change_approval',
                messageType: 'std_msgs/Bool'
            });
            var str = new ROSLIB.Message({
                data: true
            });
            pub.publish(str);
        }
    }
    LaneChangeApprovalPublisher.init();

    window.onload = function() {};
    window.onunload = function() {
        LaneChangeApprovalPublisher.ros.close();
    };
}
if (!LaneChangeApprovalStateSubscriber) {
    var LaneChangeApprovalStateSubscriber = {
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
                name: '/lane_change_approval',
                messageType: 'std_msgs/Bool'
            });
            sub.subscribe(function(message) {
                const div = document.getElementById("lane_change_approval_status");
                if (div.hasChildNodes()) {
                    div.removeChild(div.firstChild);
                }
                var res = message.data;
                var el = document.createElement("span");
                el.innerHTML = res
                document.getElementById("lane_change_approval_status").appendChild(el);
            });
        }
    }
    LaneChangeApprovalStateSubscriber.init();

    window.onload = function() {};
    window.onunload = function() {
        LaneChangeApprovalStateSubscriber.ros.close();
    };
}