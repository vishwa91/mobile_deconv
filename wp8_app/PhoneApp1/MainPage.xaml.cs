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
         * 3. Bluetooth: For sending data to the computer.
         */
        // Create a camera instance
        PhotoCamera app_camera;
        // Create a media library. Will remove this once we can write directly to bluetooth stream
        MediaLibrary photo_library = new MediaLibrary();

        // Create an instance of gyroscope and accelerometer. Once we are comfortable with accelerometer alone, we can 
        // make gyroscope optional.
        Accelerometer app_accelerometer;
        Gyroscope app_gyroscope;
        // Create timers for sampling
        DispatcherTimer accel_timer, gyro_timer;
        // A bool variable to check if data is valid
        bool isDataValid;
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
                update_info("Got a working camera. Initializing.", UpdateType.DebugSection);
                app_camera = new Microsoft.Devices.PhotoCamera(CameraType.Primary);
                // Attach event handlers.
                update_info("Attaching event handlers.", UpdateType.DebugSection);
                app_camera.Initialized += new EventHandler<CameraOperationCompletedEventArgs>(cam_initialized); // Initialized.
                app_camera.CaptureCompleted += new EventHandler<CameraOperationCompletedEventArgs>(cam_captured); // Captured.
                app_camera.CaptureImageAvailable += new EventHandler<ContentReadyEventArgs>(cam_available);   // Picture available.
                app_camera.CaptureThumbnailAvailable += new EventHandler<ContentReadyEventArgs>(cam_thumbnail); // Thumbnail available.
                app_camera.AutoFocusCompleted += new EventHandler<CameraOperationCompletedEventArgs>(cam_autofocus); // Autofocus.

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
            
        }

        protected override void OnNavigatingFrom(System.Windows.Navigation.NavigatingCancelEventArgs e)
        {
            this.Dispatcher.BeginInvoke(delegate()
            {
                txtDebug.Text = "Navigating away from the main page.";
            });
        }
        
        // Method for easy printing to screen
        void update_info(string str, UpdateType update_type)
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
               update_info("Camera initialized.", UpdateType.DebugSection);
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
                update_info("Saving image.", UpdateType.DebugSection);
                // Save picture to camera roll.
                photo_library.SavePictureToCameraRoll("savefig.jpg", e.ImageStream);
                // Done saving.
                update_info("Image saved.", UpdateType.DebugSection);
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
            update_info("Ignoring thumbnail.", UpdateType.DebugSection);
            e.ImageStream.Close();
        }
        void cam_autofocus(object sender, CameraOperationCompletedEventArgs e)
        {
           update_info("Autofocus complete.", UpdateType.DebugSection);
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
                    // Do nothing as of now.
                }
            }
        }
        private void OnButtonPress(object sender, EventArgs e)
        {
            // When button is pressed, focus first and then take a snap.
            if (app_camera != null)
            {
                // Try to focus first
                try
                {
                    app_camera.Focus();
                }
                catch (Exception focusError)
                {
                    // Don't do anything.
                }
                // (Hopefull) done with focussing. Take an image now.
                app_camera.CaptureImage();
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


    }
}