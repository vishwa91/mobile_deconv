// Main source file.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Navigation;
using Microsoft.Phone.Controls;
using Microsoft.Phone.Shell;
using PhoneApp1.Resources;
using Microsoft.Devices.Sensors; // From sensors app
using System.Windows.Threading;  // From sensors app
// Directives. From camera app
using Microsoft.Devices;        // From camera app

using Microsoft.Xna.Framework;
using System.Windows.Media.Imaging;
using Windows.Storage.Streams;
using System.Text;

// Communication module
using PhoneApp1.modules;
enum UpdateType
{
    Information,
    DebugSection,
    MessageBox
}
namespace PhoneApp1
{
    public partial class MainPage : PhoneApplicationPage
    {
        /* For this application, we will need a total of 3 broad components:
         * 1. Camera: For taking the pictures
         * 2. Sensors: For recording the motion of the camera during shutter open
         * 3. TCP: For sending data to the computer.
         */
        // Create a camera instance
        AppCamera app_camera;
        bool transmit = false;
        // Create sensors instances
        AppAccelerometer accelerometer;
        AppGyroscope gyroscope;
        DispatcherTimer accel_timer;
        // Lists for saving acceleration values during shutter open.
        List<float> accelX = new List<float>();
        List<float> accelY = new List<float>();
        List<float> accelZ = new List<float>();

        // Bool variable to enable logging.
        bool accel_log = false;
        // Constants
        const int port = 1991; 
        const string hostname = "10.21.2.208";
        // Create a com socket object
        ComSocket app_comsocket = null;
        // Constructor
        public MainPage()
        {
            InitializeComponent();
        }

        // I think OnNavigateTo is the function that is executed once the app is opened
        protected override void OnNavigatedTo(System.Windows.Navigation.NavigationEventArgs e)
        {
            // Create a new camera instance.
            app_camera = new AppCamera();
            app_camera.initialise();
            // Start the accelerometer and gyroscope service.
            accelerometer = new AppAccelerometer();
            gyroscope = new AppGyroscope();
            DeviceStatus deviceStatus;
            deviceStatus = accelerometer.start();
            if (deviceStatus == DeviceStatus.DEVICE_ERROR)
            {
                Log("Error in accelerometer device", UpdateType.DebugSection);
            }
            else
            {
                accel_timer = new DispatcherTimer();
                accel_timer.Interval = TimeSpan.FromMilliseconds(40);
                accel_timer.Tick += new EventHandler(accelerometer_timer);
                accel_timer.Start();
            }
            deviceStatus = gyroscope.start();
                      
        }

        protected override void OnNavigatingFrom(System.Windows.Navigation.NavigatingCancelEventArgs e)
        {
            this.Dispatcher.BeginInvoke(delegate()
            {
                txtDebug.Text = "Navigating away from the main page.";
            });
        }
        
        // Method for easy printing to screen
        void clrLog(UpdateType update_type)
        {
            if (update_type == UpdateType.DebugSection)
            {
                Deployment.Current.Dispatcher.BeginInvoke(delegate()
                {
                    txtDebug.Text = "";
                });
            }
            else
            {
                Deployment.Current.Dispatcher.BeginInvoke(delegate()
                {
                    txtInfo.Text = "";
                });
            }
        }
        void Log(string str, UpdateType update_type)
        {
            if (update_type == UpdateType.DebugSection)
            {
                Deployment.Current.Dispatcher.BeginInvoke(delegate()
                {
                    txtDebug.Text = str;
                });
            }
            else if (update_type == UpdateType.Information)
            {
                Deployment.Current.Dispatcher.BeginInvoke(delegate()
                {
                    //txtInfo.Text += System.Environment.NewLine;
                    txtInfo.Text = str;
                });
            }
            else if (update_type == UpdateType.MessageBox)
            {
                Deployment.Current.Dispatcher.BeginInvoke(delegate()
                {
                    MessageBox.Show(str);
                });
            }
        }       

        private void ShutterButtonClick(object sender, RoutedEventArgs e)
        {
           // Capture a picture. This will happen asynchronously. 
            if (app_camera.cam_busy == true)
            {
                Log("Camera resource busy.", UpdateType.DebugSection);
            }
            else
            {
                Log("Starting camera capture", UpdateType.DebugSection);
                accelX.Clear();
                accelY.Clear();
                accelZ.Clear();
                app_camera.capture();
            }
        }
        private void SocketConn_Click(object sender, RoutedEventArgs e)
        {
            clrLog(UpdateType.Information);
            // Make sure hostname and port are given.
            if (txtHostName.Equals("Host") || txtHostName.Equals("") || txtPort.Equals("Port") || txtPort.Equals(""))
            {
                // Release message and return.
                Log("Invalid hostname or port", UpdateType.MessageBox);
            }
            else
            {
                if (app_comsocket == null)
                {
                    Log("Attempting connection.", UpdateType.Information);
                    app_comsocket = new ComSocket();
                    string result = app_comsocket.Connect(txtHostName.Text, int.Parse(txtPort.Text));
                    Log("Connection status: " + result, UpdateType.Information);
                    if (result.Equals("Success"))
                    {
                        // Good.
                        Log("Connection Established.", UpdateType.Information);
                        app_comsocket.Send("STRT\n");
                        app_comsocket.Send("ACKR\n");
                    }
                    else
                    {
                        Log("Connection Failure", UpdateType.Information);
                        app_comsocket = null;
                    }
                }
                else
                {
                    app_comsocket.Send("ACKR");
                    Log("Connection already established", UpdateType.Information);
                }
                
            }
        }
        // The accelerometer timer not only displays the acceleration data periodically, but also 
        // logs the acceleration values when the shutter is open.
        void accelerometer_timer(object sender, EventArgs e)
        {
            // Set the viewfinderBrush source
            if (app_camera.cam_open_busy == false && app_camera.source_set == false)
            {
                viewfinderBrush.SetSource(app_camera._camera);
                app_camera.source_set = true;
            }
            // Focus the camera
            if (app_camera.focus_busy == false && app_camera.cam_busy == false)
            {
                app_camera.set_focus(focus_slider.Value);
                Log(focus_slider.Value.ToString(), UpdateType.DebugSection);
            }
            Vector3 accel = accelerometer.getvalue();
            // If logging is enabled, send the data to the com socket
            if (accel_log == true)
            {
                if (app_comsocket != null)
                {
                    app_comsocket.Send(accel.X.ToString() + ";" + accel.Y.ToString() + ";" + accel.Z.ToString() + ";;");
                }
            }
            if (app_camera.cam_busy == true)
            {
                focus_slider.IsEnabled = false;
                accelX.Add(accel.X);
                accelY.Add(accel.Y);
                accelZ.Add(accel.Z);
            }
            if ((app_camera.cam_busy == false) && (app_camera.transmit == true))
            {
                Log("Camera capture complete", UpdateType.DebugSection);
                if (app_comsocket != null)
                {
                    // Send acceleration data
                    app_comsocket.Send("STAC\n");
                    string accel_string = "";
                    for (int i = 0; i < accelX.Count; i++)
                        accel_string += accelX[i].ToString() + ";" + accelY[i].ToString() + ";" + accelZ[i].ToString() + ";;";
                    accel_string += "\n";
                    app_comsocket.Send(accel_string);
                    app_comsocket.Send("EDAC\n");
                    
                    // Send image data length
                    byte[] imarray = app_camera.imstream.ToArray();
                    app_comsocket.Send("STIL\n");
                    app_comsocket.Send(imarray.Length.ToString() + '\n');
                    app_comsocket.Send("EDIL\n");
                    // Send focus details
                    app_comsocket.Send("STFC\n");
                    app_comsocket.Send(focus_slider.Value.ToString()+"\n");
                    app_comsocket.Send("EDFC\n");
                    // Send image data
                    app_comsocket.Send("STIM\n"); 
                    string imstring = Encoding.Unicode.GetString(imarray, 0, imarray.Length);
                    app_comsocket.Send(imarray);
                    app_comsocket.Send("\n");
                    Log("Image size is " + app_camera.imheight.ToString()+";"+app_camera.imwidth.ToString(), UpdateType.DebugSection);

                    app_comsocket.Send("\n");
                     
                    app_comsocket.Send("EDIM\n");                    
                    // Done.
                    app_comsocket.Send("ENDT\n");
                    app_comsocket.Close();
                    app_comsocket = null;
                }
                Log("Total readings: " + accelX.Count.ToString(), UpdateType.Information);
                app_camera.transmit = false;
                focus_slider.IsEnabled = true;
            }
            Deployment.Current.Dispatcher.BeginInvoke(delegate()
            {
                txtAccel.Text = string.Format("x:{0}\n,y:{1}\n,z:{2}", accel.X.ToString("0.00"), accel.Y.ToString("0.00"), accel.Z.ToString("0.00"));
            });
        }

        private void start_log(object sender, RoutedEventArgs e)
        {
            if (accel_log == false)
            {
                Log("Starting accelerometer logging. Please keep the phone in static state.", UpdateType.MessageBox);
                accel_log = true;
                sensor_button.Content = "Stop sensor log";
                if (app_comsocket != null)
                    app_comsocket.Send("STLG\n");
            }
            else
            {
                if (app_comsocket != null)
                    app_comsocket.Send("\nEDLG\n");
                accel_log = false;
                sensor_button.Content = "Start sensor log";
            }
        }
    }
    
}