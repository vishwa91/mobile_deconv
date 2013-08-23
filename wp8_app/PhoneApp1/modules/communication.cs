// Source file for communication routines.
// Currently, communication is expected to be done using socket communication.

// System includes
using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Text;

namespace PhoneApp1.modules
{
    class ComSocket
    {
        static ManualResetEvent _clientDone = new ManualResetEvent(false);  //Notifies completion of asynchronous call.
        const int TIMEOUT_IN_MILLISECONDS = 5000;  // Timeout in case of failed asynchronous call.
        const int MAX_BUFFER_SIZE = 2048;   // As of now, we won't send image. Let us experiment first.

        // Method to connect to the remote server.
        public string Connect(string hostname, int port)
        {
            string result = string.Empty; // Hold result of connection attempt.
            DnsEndPoint hostentry = new DnsEndPoint(hostname, port);
            _socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            SocketAsyncEventArgs socketEventArg = new SocketAsyncEventArgs(); // Used for connecting async tcp connection.
            socketEventArg.RemoteEndPoint = hostentry;
            socketEventArg.Completed += new EventHandler<SocketAsyncEventArgs>(delegate(object s, SocketAsyncEventArgs e)
            {
                // Retrieve the result of this request
                result = e.SocketError.ToString();

                // Signal that the request is complete, unblocking the UI thread
                _clientDone.Set();
            });
            _clientDone.Reset();    // Done. Reset the client.
            _socket.ConnectAsync(socketEventArg);   // Send an asynchronous request.
            _clientDone.WaitOne(TIMEOUT_IN_MILLISECONDS); // Wait for some time for connection to succeed. 

            // Return the result.
            return result;
        }
        // Method to send data to the remote server.
        public string Send(string data)
        {
            string response = "Operation timeout.";
            // Hopefully, _socket is not null;
            if (_socket != null)
            {
                // SocketAsyncEventArgs is used for sending the event arguments while communicating.
                SocketAsyncEventArgs socketEventArg = new SocketAsyncEventArgs();

                // Set the remote server
                socketEventArg.RemoteEndPoint = _socket.RemoteEndPoint;
                socketEventArg.UserToken = null; // What is this?
                // Handler for completed transaction
                socketEventArg.Completed += new EventHandler<SocketAsyncEventArgs>(delegate(object s, SocketAsyncEventArgs e)
                    {
                        response = e.SocketError.ToString();
                        _clientDone.Set(); // Done. Set UI thread free.
                    });
                // Create encoded data to send to the remote server.
                byte[] data_out = Encoding.Unicode.GetBytes(data);
                socketEventArg.SetBuffer(data_out, 0, data_out.Length);

                // Done sending data.
                _clientDone.Reset();
                // Now send the data.
                _socket.SendAsync(socketEventArg);
                // Wait for some time to see if there is a timeout.
                _clientDone.WaitOne(TIMEOUT_IN_MILLISECONDS);
            }
            else
            {
                // Socket not created.
                response = "Socket not initialized.";
            }
            return response;
        }
        // Method to receive data from the remote server.
        public string Receive()
        {
            string response = "Operation timeout.";
            // Hopefully, _socket is not null;
            if (_socket != null)
            {
                // SocketAsyncEventArgs is used for sending the event arguments while communicating.
                SocketAsyncEventArgs socketEventArg = new SocketAsyncEventArgs();

                // Set the remote server
                socketEventArg.RemoteEndPoint = _socket.RemoteEndPoint;
                // Set the data buffer to receive data
                socketEventArg.SetBuffer(new Byte[MAX_BUFFER_SIZE], 0, MAX_BUFFER_SIZE);
                // Handler for completed transaction
                socketEventArg.Completed += new EventHandler<SocketAsyncEventArgs>(delegate(object s, SocketAsyncEventArgs e)
                {
                    if (e.SocketError == SocketError.Success)
                    {
                        response = Encoding.UTF8.GetString(e.Buffer, e.Offset, e.BytesTransferred);
                        response = response.Trim('\0');
                    }
                    else
                    {
                        response = e.SocketError;
                    }
                    _clientDone.Set(); // Done. Set UI thread free.
                });
               
                // Done sending data.
                _clientDone.Reset();
                // Now send the data.
                _socket.ReceiveAsync(socketEventArg);
                // Wait for some time to see if there is a timeout.
                _clientDone.WaitOne(TIMEOUT_IN_MILLISECONDS);
            }
            else
            {
                // Socket not created.
                response = "Socket not initialized.";
            }
            return response;
        }
        // Method to close the connection
        public void Close()
        {
            if (_socket != null)
                _socket.Close();
        }
    }
}