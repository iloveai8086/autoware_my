Header header
sensor_msgs/NavSatFix   gnss_info

# float64 latitude        # in Degree
# float64 longitude       # in Degree
# float64 altitude        # in Degree
float32 roll            # in Degree
float32 pitch           # in Degree
float32 yaw             # in Degree

# float32 speed_north
# float32 speed_east
# float32 speed_earth

# float32[9] errors       # the error of Above 9 data, in Meter or m/s
# float32[9] position_covariance  # provide by slam


# float32 acceleration_north
# float32 acceleration_east
# float32 acceleration_earth

# float32 angular_velocity_roll
# float32 angular_velocity_pitch
# float32 angular_velocity_yaw

int8 status 
float32 result_score