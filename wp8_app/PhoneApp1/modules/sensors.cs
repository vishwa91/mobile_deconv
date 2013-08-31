// Source file for sensors data.
using System;
using System.Windows;
using Microsoft.Phone.Controls;

using Microsoft.Devices.Sensors;
using Microsoft.Xna.Framework;
using System.Windows.Threading;

public enum DeviceStatus
{
    DEVICE_OK,
    DEVICE_ERROR,
};

namespace PhoneApp1.modules
{
    // we will have two classes, one for accelerometer and one for gyroscope
    public class AppAccelerometer
    {
        // Create an instance of an accelerometer
        Accelerometer _accelerometer;
        DispatcherTimer _timer;
        Vector3 acceleration;
        bool isDataValid;

        public AppAccelerometer()
        {
            // Check if we have accelerometer support
            if (Accelerometer.IsSupported)
            {
                // Create a dispatch timer to regularly poll the data.
                _timer = new DispatcherTimer();
                _timer.Interval = TimeSpan.FromMilliseconds(20);
                _timer.Tick += new EventHandler(_timer_tick);

                // Create an accelerometer
                _accelerometer = new Accelerometer();
                // Add a poll time
                _accelerometer.TimeBetweenUpdates = TimeSpan.FromMilliseconds(20);
                // Attach an event handler
                _accelerometer.CurrentValueChanged += new EventHandler<SensorReadingEventArgs<AccelerometerReading>>(accel_curval_changed);
                MessageBox.Show("Found accelerometer.");
            }
            else
            {
                MessageBox.Show("Accelerometer not found");
                _accelerometer = null;
            }
        }
        public DeviceStatus start()
        {
            // Start accelerometer only if it works
            if ( _accelerometer != null)
            {
                _accelerometer.Start();
                _timer.Start();
                return DeviceStatus.DEVICE_OK;
            }
            else
                return DeviceStatus.DEVICE_ERROR;
        }
        public DeviceStatus stop()
        {
            if (_accelerometer != null)
            {
                _accelerometer.Stop();
                _timer.Stop();
                return DeviceStatus.DEVICE_OK;
            }
            else
                return DeviceStatus.DEVICE_ERROR;
        }
        public Vector3 getvalue()
        {
            return acceleration;
        }

        void accel_curval_changed(object sender, SensorReadingEventArgs<AccelerometerReading> e)
        {
            // Update member values
            isDataValid = _accelerometer.IsDataValid;
            acceleration = e.SensorReading.Acceleration;
        }
        void _timer_tick(object sender, EventArgs e)
        {
            // contrary to what I thought, _timer_tick is used for updating the GUI. Do nothing now. Let us see if we need it.
        }
    }

    public class AppGyroscope
    {
        // Create an instance of the gyroscope
        Gyroscope _gyroscope;
        DispatcherTimer _timer;
        Vector3 rotation_rate;
        bool isDataValid;

        public AppGyroscope()
        {
            // Check if we have gyroscope support
            if (Gyroscope.IsSupported)
            {
                _gyroscope = new Gyroscope();
                // Add polling timer
                _gyroscope.TimeBetweenUpdates = TimeSpan.FromMilliseconds(20);
                // Add an event handler for the gyroscope.
                _gyroscope.CurrentValueChanged += new EventHandler<SensorReadingEventArgs<GyroscopeReading>>(gyro_curval_changed);
                // Create a dispatch timer. It is only for GUI support. We will get rid of it soon.
                _timer = new DispatcherTimer();
                _timer.Interval = TimeSpan.FromMilliseconds(20);
                _timer.Tick += new EventHandler(timer_tick);
            }
            else
            {
                _gyroscope = null;
                MessageBox.Show("Gyroscope not supported.");
            }
        }
        public DeviceStatus start()
        {
            if (_gyroscope != null)
            {
                _gyroscope.Start();
                _timer.Start();
                return DeviceStatus.DEVICE_OK;
            }
            else
                return DeviceStatus.DEVICE_ERROR;
        }
        public DeviceStatus stop()
        {
            if (_gyroscope != null)
            {
                _gyroscope.Stop();
                _timer.Stop();
                return DeviceStatus.DEVICE_OK;
            }
            else
                return DeviceStatus.DEVICE_ERROR;
        }
        public Vector3 getvalue()
        {
            return rotation_rate;
        }
        void gyro_curval_changed(object sender, SensorReadingEventArgs<GyroscopeReading> e)
        {
            isDataValid = _gyroscope.IsDataValid;
            rotation_rate = e.SensorReading.RotationRate;
        }
        void timer_tick(object sender, EventArgs e)
        {
            // nothing right now.
        }
    }
}