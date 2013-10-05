// Source file for camera device
using System;
using System.Windows;
using System.Collections.Generic;
using Microsoft.Phone.Controls;
using System.IO;

using System.Windows.Media;
using Windows.Phone.Media.Capture;
using Microsoft.Xna.Framework.Media;

using PhoneApp1.Resources;

namespace PhoneApp1.modules
{
    class AppCamera
    {
        public PhotoCaptureDevice _camera = null;
        public CameraCaptureSequence _camsequence = null;
        public MemoryStream imstream = null;
        public bool cam_open_busy, cam_busy, transmit;
        public bool source_set = false, focus_busy=false;
        public int imheight, imwidth;
        public UInt32 focus_min, focus_max;
        public async void initialise()
        {
            // Disable transmit.
            transmit = false;
            // Get available resolutions.
            IReadOnlyList<Windows.Foundation.Size> available_res = PhotoCaptureDevice.GetAvailableCaptureResolutions(CameraSensorLocation.Back);
            int count = available_res.Count;
            // Make the resolution details public
            imheight = (int)available_res[count-1].Height;
            imwidth = (int)available_res[count-1].Width;
            // Open a new capture device asynchronously.    
            cam_open_busy = true;
            _camera = await PhotoCaptureDevice.OpenAsync(CameraSensorLocation.Back, available_res[count-1]);
            cam_open_busy = false;
            // Set the exposure time to 1s
            _camera.SetProperty(KnownCameraPhotoProperties.ExposureTime, 1000000);
            // Create a new sequence
            _camsequence = _camera.CreateCaptureSequence(1);
            // Create a new memory stream.
            imstream = new MemoryStream();
            _camsequence.Frames[0].CaptureStream = imstream.AsOutputStream();
            // Wait for the camera to initialize.
            await _camera.PrepareCaptureSequenceAsync(_camsequence);
        }
        public async void capture()
        {
            // Take a picture. Flag busy meanwhile.
            cam_busy = true;
            await _camsequence.StartCaptureAsync();
            cam_busy = false;
            transmit = true;
            imstream.Seek(0, SeekOrigin.Begin);
        }
        public async void set_focus(double focus_val)
        {
            focus_busy = true;
            try
            {
                CameraCapturePropertyRange range = PhotoCaptureDevice.GetSupportedPropertyRange(CameraSensorLocation.Back, KnownCameraGeneralProperties.ManualFocusPosition);
                //double value = (UInt32)range.Min;
                double value = (UInt32)range.Min + (focus_val / 100.0) * ((UInt32)range.Max - (UInt32)range.Min);
                focus_min = (UInt32)range.Min;
                focus_max = (UInt32)range.Max;
                _camera.SetProperty(KnownCameraGeneralProperties.ManualFocusPosition, (UInt32)value);
            }
            //   
            catch (Exception err)
            {
                MessageBox.Show(err.ToString());
            }
            await _camera.FocusAsync();
            focus_busy = false;
        }
    }
}