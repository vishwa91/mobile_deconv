// Main source file.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents; // From camera app
using System.Windows.Input;     // From camera app
using System.Windows.Media;     // From camera app
using System.Windows.Media.Animation; // From camera app
using System.Windows.Shapes;    // From camera app
using System.Windows.Navigation;
using Microsoft.Phone.Controls;
using Microsoft.Phone.Shell;
using PhoneApp1.Resources;
using Microsoft.Devices.Sensors; // From sensors app
using System.Windows.Threading;  // From sensors app
// Directives. From camera app
using Microsoft.Devices;        // From camera app
using System.IO;                // From camera app
using System.IO.IsolatedStorage;// From camera app
using Microsoft.Xna.Framework.Media; // From camera app
using Microsoft.Xna.Framework;
using System.Windows.Media.Imaging;
using Windows.Storage.Streams;

// Communication module
using PhoneApp1.modules;
enum UpdateType
{
    Information,
    DebugSection
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
        PhotoCamera app_camera;
        // Resolution object.
        Size resolution = new Size();

        // Create a media library. Will remove this once we can write directly to bluetooth stream
        MediaLibrary photo_library = new MediaLibrary();

        // Create sensors instances
        AppAccelerometer accelerometer;
        AppGyroscope gyroscope;
        DispatcherTimer accel_timer;
        // Lists for saving acceleration values during shutter open.
        List<float> accelX = new List<float>();
        List<float> accelY = new List<float>();
        List<float> accelZ = new List<float>();
        bool shutter_open = true;
        // Constants
        const int port = 1991; // Port number is not our wish. 7 is an echo server
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
            // Check if we have a camera at the back. Ignore any front cameras.
           
            if (PhotoCamera.IsCameraTypeSupported(CameraType.Primary) == true)
            {
                Log("Got a working camera. Initializing.", UpdateType.DebugSection);
                app_camera = new Microsoft.Devices.PhotoCamera(CameraType.Primary);
                // Attach event handlers.
                Log("Attaching event handlers.", UpdateType.DebugSection);
                app_camera.Initialized += new EventHandler<CameraOperationCompletedEventArgs>(cam_initialized); // Initialized.
                app_camera.CaptureCompleted += new EventHandler<CameraOperationCompletedEventArgs>(cam_captured); // Captured.
                app_camera.CaptureImageAvailable += new EventHandler<ContentReadyEventArgs>(cam_available);   // Picture available.
                app_camera.CaptureThumbnailAvailable += new EventHandler<ContentReadyEventArgs>(cam_thumbnail); // Thumbnail available.
                app_camera.AutoFocusCompleted += new EventHandler<CameraOperationCompletedEventArgs>(cam_autofocus); // Autofocus.
                resolution = app_camera.Resolution;
                
                // Instead of using a button, we need to figure out a way to send a remote request from the PC
                // But currently, just use a button
                CameraButtons.ShutterKeyPressed += OnButtonPress;
                CameraButtons.ShutterKeyReleased += OnButtonRelease;
                // Set the source for the view finder canvas
                viewfinderBrush.SetSource(app_camera);
            }
            else
            {
                this.Dispatcher.BeginInvoke(delegate()
                {
                    txtDebug.Text = "Camera not available.";
                });
                ShutterButton.IsEnabled = false;
            }
            // Start the accelerometer and gyroscope service.
            accelerometer = new AppAccelerometer();
            gyroscope = new AppGyroscope();
            DeviceStatus deviceStatus;
            deviceStatus = accelerometer.start();
            if (deviceStatus == DeviceStatus.DEVICE_ERROR)
            {
                Deployment.Current.Dispatcher.BeginInvoke(delegate()
                {
                    txtDebug.Text = "Error in accelerometer device";
                });
            }
            else
            {
                accel_timer = new DispatcherTimer();
                accel_timer.Interval = TimeSpan.FromMilliseconds(20);
                accel_timer.Tick += new EventHandler(accelerometer_timer);
                accel_timer.Start();
            }
            deviceStatus = gyroscope.start();
            /*
            if (deviceStatus == DeviceStatus.DEVICE_ERROR)
            {
                Deployment.Current.Dispatcher.BeginInvoke(delegate()
                {
                    txtDebug.Text = "Error in gyroscope device"+;
                });
            }*/
            
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
            else
            {
                Deployment.Current.Dispatcher.BeginInvoke(delegate()
                {
                    //txtInfo.Text += System.Environment.NewLine;
                    txtInfo.Text = str;
                });
            }
        }       
        // Define the camera event handlers
        void cam_initialized(object sender, Microsoft.Devices.CameraOperationCompletedEventArgs e)
        {
            // Check if the initialization has succeeded.
            if (e.Succeeded)
            {
               Log("Camera initialized.", UpdateType.DebugSection);
            }
        }
        void cam_captured(object sender, CameraOperationCompletedEventArgs e)
        {
            // Nothing to do here. Sending data over bluetooth will be later taken 
            // care in the CaptureImage Available function.
        }
        void cam_available(object sender, Microsoft.Devices.ContentReadyEventArgs e)
        {
            // We will need to send the image over bluetooth later. As of now, just save.
            try
            {
                Log("Saving image.", UpdateType.DebugSection);
                long imsize = e.ImageStream.Length;
                byte[] imbuffer = new byte[imsize];
                e.ImageStream.Read(imbuffer, 0, imbuffer.Length);
                if (app_comsocket != null)
                {
                    app_comsocket.Send("AVLC:" + accelX.Count + "\n");
                    app_comsocket.Send("SIZE:" + imsize + "\n");
                    app_comsocket.Send("HGHT:" + resolution.Height + "\n");
                    app_comsocket.Send("WDTH:" + resolution.Width + "\n");
                    app_comsocket.Send("STIM" + "\n");
                    string imstring = null;
                    Log("Expecting " + ((imsize / 1024)).ToString() + "Packets", UpdateType.DebugSection);
                    imstring = System.Text.Encoding.Unicode.GetString(imbuffer, 0, (int)imsize);
                    app_comsocket.Send(imstring + "\n");
                    app_comsocket.Send("ENDT\n");
                }

                // Done saving.
                Log("Image saved.", UpdateType.DebugSection);
                e.ImageStream.Seek(0, SeekOrigin.Begin);
                // No saving to isolated store right now.
            }
            finally
            {
                // Close the image stream.
                e.ImageStream.Close();
            }
        }
        public void cam_thumbnail(object sender, ContentReadyEventArgs e)
        {
            /*
            //Log("Ignoring thumbnail.", UpdateType.DebugSection);
            Log("Sending thumbnail", UpdateType.DebugSection);
            long imsize = e.ImageStream.Length;
            byte[] imbuffer = new byte[imsize];
            e.ImageStream.Read(imbuffer, 0, imbuffer.Length);
            if (app_comsocket != null)
            {
                app_comsocket.Send("SIZE:" + imsize + "\n");
                app_comsocket.Send("HGHT:" + resolution.Height + "\n");
                app_comsocket.Send("WDTH:" + resolution.Width + "\n");
                app_comsocket.Send("STIM" + "\n");
                string imstring = null;
                Log("Expecting " + ((imsize / 1024)).ToString() + "Packets", UpdateType.DebugSection);
                imstring = System.Text.Encoding.Unicode.GetString(imbuffer, 0, (int)imsize);
                app_comsocket.Send(imstring + "\n");
               
                string substring = null;
                for (int i = 0; i < (int)(imsize / 1024); i++)
                {
                    Log("Sending packet " + i.ToString(), UpdateType.Information);
                    try
                    {
                        //imstring = System.Text.Encoding.Unicode.GetString(imbuffer, i * 1024, (i + 1) * 1024);
                        substring = imstring.Substring(i*1024, 1024);
                    }
                    catch (Exception overflow)
                    {
                        Deployment.Current.Dispatcher.BeginInvoke(delegate()
                        {
                            MessageBox.Show(string.Format("{0}, {1}", imsize, i * 1024));
                        });
                    }
                    app_comsocket.Send(substring);
                }
                app_comsocket.Send("ENDT\n");
            }

            // Done saving.
            Log("Image saved.", UpdateType.DebugSection);
            e.ImageStream.Seek(0, SeekOrigin.Begin);
            e.ImageStream.Close();*/
        }
        void cam_autofocus(object sender, CameraOperationCompletedEventArgs e)
        {
           Log("Autofocus complete.", UpdateType.DebugSection);
            // This is where we start our data aquisition from the accelerometer.
            // Every, say 20ms, we need to poll the sensor, save the data in an
            // array and send the data along with the image.
        }
        // Button gestures
        // Function from XAML script. Not sure of it's functionality
        private void ShutterButtonClick(object sender, RoutedEventArgs e)
        {
            if (app_camera != null)
            {
                // Try an image capture
                try
                {
                    txtDebug.Text = "Trying to get an image.";
                    app_camera.CaptureImage();
                    txtDebug.Text = "Got an image.";
                }
                catch (Exception ex)
                {
                    Log(ex.ToString(), UpdateType.DebugSection);
                }
            }
        }
        private void OnButtonPress(object sender, EventArgs e)
        {
            // When button is pressed, focus first and then take a snap.
            if (app_camera != null)
            {
                // Try to focus first
                shutter_open = true;
                try
                {
                    app_camera.Focus();
                }
                catch (Exception focusError)
                {
                    Log(focusError.ToString(), UpdateType.DebugSection);
                }
                // (Hopefull) done with focussing. Take an image now.                
                app_camera.CaptureImage();
                shutter_open = false;
            }
        }
        private void OnButtonRelease(object sender, EventArgs e)
        {
            // Done capturing, release focus.
            if (app_camera != null)
            {
                app_camera.CancelFocus();
            }
        }       
        private void SocketConn_Click(object sender, RoutedEventArgs e)
        {
            clrLog(UpdateType.Information);
            // Make sure hostname and port are given.
            if (txtHostName.Equals("Host") || txtHostName.Equals("") || txtPort.Equals("Port") || txtPort.Equals(""))
            {
                // Release message and return.
                Deployment.Current.Dispatcher.BeginInvoke(delegate()
                {
                    MessageBox.Show("Invalid hostname or port.");
                });
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
        void accelerometer_timer(object sender, EventArgs e)
        {
            Vector3 accel = accelerometer.getvalue();
            if (shutter_open == true)
            {
                accelX.Add(accel.X);
                accelY.Add(accel.Y);
                accelZ.Add(accel.Z);
            }
            Deployment.Current.Dispatcher.BeginInvoke(delegate()
            {
                txtAccel.Text = string.Format("x:{0}\n,y:{1}\n,z:{2}", accel.X.ToString("0.00"), accel.Y.ToString("0.00"), accel.Z.ToString("0.00"));
            });
        }

    }
    
}