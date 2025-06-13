from OpenSSL import crypto
import socket
import os

def get_local_ip():
    """Get the local IP address of the machine."""
    try:
        # Get the local hostname
        hostname = socket.gethostname()
        # Get the IP address
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except Exception as e:
        print(f"Error getting local IP: {str(e)}")
        return "127.0.0.1"

def generate_self_signed_cert():
    # Generate key
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 2048)

    # Generate certificate
    cert = crypto.X509()
    cert.get_subject().C = "IN"
    cert.get_subject().ST = "Tamil Nadu"
    cert.get_subject().L = "Coimbatore"
    cert.get_subject().O = "UYIR"
    cert.get_subject().OU = "Development"
    cert.get_subject().CN = "localhost"
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365*24*60*60)  # Valid for one year
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha256')

    # Write certificate
    with open("cert.pem", "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    
    # Write private key
    with open("key.pem", "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))

    print("Generated SSL certificates: cert.pem and key.pem")

if __name__ == "__main__":
    generate_self_signed_cert() 