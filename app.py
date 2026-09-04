from flask import Flask, flash, render_template, request, redirect, url_for, session, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import random
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import smtplib
from email.message import EmailMessage
from pathlib import Path

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-only-secret-key-change-me')

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    print('Warning: Supabase credentials are not configured. Database features will not work.')

# Email configuration - menggunakan Gmail SMTP
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', '587'))
EMAIL_USERNAME = os.getenv('EMAIL_USERNAME')
EMAIL_APP_PASSWORD = os.getenv('EMAIL_APP_PASSWORD')

# Load model and scaler
try:
    BASE_DIR = Path(__file__).resolve().parent
    model = joblib.load(BASE_DIR / 'model.pkl')
    scaler = joblib.load(BASE_DIR / 'scaler.pkl')
    print("Model dan scaler berhasil dimuat")
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    model = None
    scaler = None

def get_time_category():
    """
    Mengkategorikan waktu berdasarkan jam real-time:
    Pagi: 05:00-10:59 (encoding = 1)
    Siang: 11:00-14:59 (encoding = 2) 
    Sore: 15:00-18:59 (encoding = 3)
    Malam: 19:00-04:59 (encoding = 0)
    """
    now = datetime.now()
    hour = now.hour
    
    if 5 <= hour <= 10:
        return 1  # Pagi
    elif 11 <= hour <= 14:
        return 2  # Siang
    elif 15 <= hour <= 18:
        return 3  # Sore
    else:
        return 0  # Malam

def get_account_balance(nomor_rekening):
    """Mengambil saldo rekening dari database"""
    try:
        response = supabase.table('rekening').select('account_balance').eq('nomor_rekening', nomor_rekening).execute()
        if response.data:
            return float(response.data[0]['account_balance'])
        return 0.0
    except Exception as e:
        print(f"Error getting account balance: {e}")
        return 0.0

def process_transaction_new_model(amount, sender_account, recipient_account):
    """
    Process transaction with new ML model using updated variables:
    - amount: jumlah transaksi
    - oldBalInitiator: saldo pengirim sebelum transaksi
    - newBalInitiator: saldo pengirim sesudah transaksi
    - oldBalRecipient: saldo penerima sebelum transaksi
    - newBalRecipient: saldo penerima sesudah transaksi
    - isFlaggedFraud: 1 jika transaksi > 3.2 miliar, 0 jika tidak
    - waktu: kategori waktu (0=malam, 1=pagi, 2=siang, 3=sore)
    
    Uses threshold 0.8 for fraud detection
    """
    try:
        # Ambil saldo sebelum transaksi (oldBal)
        oldBalInitiator = get_account_balance(sender_account)
        oldBalRecipient = get_account_balance(recipient_account)
        
        # Hitung saldo sesudah transaksi (newBal)
        newBalInitiator = oldBalInitiator - amount
        newBalRecipient = oldBalRecipient + amount
        
        # Tentukan isFlaggedFraud (1 jika amount > 3.2 miliar)
        isFlaggedFraud = 1 if amount > 3200000000 else 0
        
        # Dapatkan kategori waktu real-time
        waktu = get_time_category()
        
        # Prepare features for new model
        features_data = {
            'amount': amount,
            'oldBalInitiator': oldBalInitiator,
            'newBalInitiator': newBalInitiator,
            'oldBalRecipient': oldBalRecipient,
            'newBalRecipient': newBalRecipient,
            'isFlaggedFraud': isFlaggedFraud,
            'waktu': waktu
        }
        
        print(f"Transaction features: {features_data}")
        
        # Kolom numerical yang perlu di-scaling
        numerical_cols = ['amount', 'oldBalInitiator', 'newBalInitiator', 'oldBalRecipient', 'newBalRecipient']
        
        # Buat DataFrame untuk preprocessing
        df = pd.DataFrame([features_data])
        
        # Scale numerical columns menggunakan scaler.pkl
        if scaler is not None:
            df[numerical_cols] = scaler.transform(df[numerical_cols])
            print(f"Scaled features: {df.iloc[0].to_dict()}")
        else:
            print("Warning: Scaler not loaded, using raw values")
        
        # Siapkan input untuk model dengan 7 variabel yang tepat
        # Urutan: amount, oldBalInitiator, newBalInitiator, oldBalRecipient, newBalRecipient, isFlaggedFraud, waktu
        model_input = np.array([[
            df['amount'].iloc[0],                    # amount (scaled)
            df['oldBalInitiator'].iloc[0],          # oldBalInitiator (scaled)
            df['newBalInitiator'].iloc[0],          # newBalInitiator (scaled)
            df['oldBalRecipient'].iloc[0],          # oldBalRecipient (scaled)
            df['newBalRecipient'].iloc[0],          # newBalRecipient (scaled)
            features_data['isFlaggedFraud'],        # isFlaggedFraud (binary, tidak di-scale)
            features_data['waktu']                  # waktu (categorical, tidak di-scale)
        ]])
        
        print(f"Model input: {model_input}")
        
        # Prediksi menggunakan model dengan threshold 0.8
        if model is not None:
            # Get probability for fraud class (class 1)
            fraud_probability = model.predict_proba(model_input)[0][1]
            probability_percent = fraud_probability * 100
            
            # Apply threshold 0.8 for fraud detection
            FRAUD_THRESHOLD = float(os.getenv('FRAUD_THRESHOLD', '0.8'))
            prediction = 1 if fraud_probability >= FRAUD_THRESHOLD else 0
            
            print(f"Fraud probability: {fraud_probability:.4f} ({probability_percent:.2f}%)")
            print(f"Threshold: {FRAUD_THRESHOLD}")
            print(f"Final prediction: {prediction} ({'FRAUD' if prediction == 1 else 'LEGITIMATE'})")
            
            return prediction, round(probability_percent, 2), features_data
        else:
            print("Warning: Model not loaded, returning default values")
            return 0, 0.0, features_data
            
    except Exception as e:
        print(f"Error in process_transaction_new_model: {e}")
        # Return default values in case of error
        return 0, 0.0, {}

def send_email_otp(recipient_email, otp_code, recipient_name):
    """Send OTP via email menggunakan Gmail SMTP - berdasarkan test_email.py"""
    try:
        print(f"Attempting to send OTP {otp_code} to {recipient_email}")
        
        # Setup SMTP server
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        
        # Login ke Gmail menggunakan app password
        if not EMAIL_USERNAME or not EMAIL_APP_PASSWORD:
            raise RuntimeError('EMAIL_USERNAME and EMAIL_APP_PASSWORD are not configured.')
        server.login(EMAIL_USERNAME, EMAIL_APP_PASSWORD)
        print("Gmail login successful!")
        
        # Create email message
        msg = EmailMessage()
        msg['Subject'] = 'FINS - Kode Verifikasi Transaksi'
        msg['From'] = EMAIL_USERNAME
        msg['To'] = recipient_email
        
        # Email body content
        email_body = f"""Halo {recipient_name},

Kami mendeteksi aktivitas transaksi yang memerlukan verifikasi tambahan untuk keamanan akun Anda.

Kode OTP Anda: {otp_code}

Kode ini akan berlaku selama 10 menit. Jangan bagikan kode ini kepada siapa pun.

Jika Anda tidak melakukan transaksi ini, segera hubungi customer service kami di support@fins.com

Terima kasih,
Tim Keamanan FINS"""
        
        msg.set_content(email_body)
        
        # Send email
        server.send_message(msg)
        server.quit()
        
        print(f"OTP {otp_code} berhasil dikirim ke {recipient_email}")
        return True
        
    except Exception as e:
        print(f"Error mengirim email: {e}")
        # Fallback ke simulation mode jika gagal
        print(f"[EMAIL SIMULATION] OTP {otp_code} untuk {recipient_email}")
        return False  # Return False jika email gagal dikirim

def get_user_by_account_id(account_id):
    """Get user data from Supabase"""
    try:
        response = supabase.table('nasabah').select('*').eq('account_id', account_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        print(f"Error fetching user: {e}")
        return None

def get_user_account(account_id):
    """Get user's rekening data from Supabase"""
    try:
        response = supabase.table('rekening').select('*').eq('account_id', account_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        print(f"Error fetching account: {e}")
        return None

def authenticate_user(account_id, password):
    """Authenticate user with Supabase"""
    try:
        response = supabase.table('nasabah').select('*').eq('account_id', account_id).eq('password', password).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        print(f"Error authenticating user: {e}")
        return None

def update_account_balance(nomor_rekening, new_balance):
    """Update account balance in Supabase"""
    try:
        response = supabase.table('rekening').update({
            'account_balance': new_balance
        }).eq('nomor_rekening', nomor_rekening).execute()
        return response.data
    except Exception as e:
        print(f"Error updating balance: {e}")
        return None

def save_transaction_to_db(transaction_data):
    """Save transaction to Supabase with proper formatting"""
    try:
        # Ensure transaction_date is in proper format
        if 'transaction_date' in transaction_data:
            if isinstance(transaction_data['transaction_date'], str):
                # If it's already a string, keep as is
                pass
            else:
                # Convert datetime to ISO format string
                transaction_data['transaction_date'] = transaction_data['transaction_date'].isoformat()
        
        print(f"Saving transaction: {transaction_data}")  # For debugging
        
        response = supabase.table('transaction').insert(transaction_data).execute()
        
        if response.data:
            print(f"Transaction saved successfully: {response.data[0]['transaction_id']}")
            return response.data
        else:
            print("Failed to save transaction - no data returned")
            return None
            
    except Exception as e:
        print(f"Error saving transaction: {e}")
        return None

def calculate_customer_age(birth_date):
    """Calculate age from birth date - TIDAK DIGUNAKAN LAGI untuk model baru"""
    try:
        if isinstance(birth_date, str):
            # Parse string date in format YYYY-MM-DD
            birth_date = datetime.strptime(birth_date, '%Y-%m-%d').date()
        elif hasattr(birth_date, 'date'):
            # If it's a datetime object, get the date part
            birth_date = birth_date.date()
        
        today = datetime.now().date()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        
        print(f"Calculated age: {age} from birth date: {birth_date}")  # For debugging
        return age
    except Exception as e:
        print(f"Error calculating age: {e}")
        return 25  # Default age if calculation fails

def format_date_for_display(date_str):
    """Format date string for display"""
    try:
        if isinstance(date_str, str):
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            return date_obj.strftime('%B %d, %Y')
        return str(date_str)
    except:
        return 'Invalid date'

@app.route('/')
def home():
    """Redirect to login page"""
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        account_id = request.form['username']  # Using account_id as username
        password = request.form['password']
        
        # Initialize or increment login attempts for this session
        if 'login_attempts' not in session:
            session['login_attempts'] = 0
        session['login_attempts'] += 1
        
        # Authenticate user
        user = authenticate_user(account_id, password)
        
        if user:
            # Get user's account/rekening data
            account = get_user_account(account_id)
            
            if account:
                # Store in session - including PIN and login attempts
                session['account_id'] = account_id
                session['user_data'] = user
                session['account_data'] = account
                session['balance'] = account['account_balance']
                session['pin'] = user.get('pin')
                # Keep login attempts for fraud detection model
                
                return redirect(url_for('dashboard'))
            else:
                return render_template('login.html', error="Account not found")
        else:
            return render_template('login.html', error="Invalid Account ID or Password")
    
    # GET request - reset login attempts for new session
    if 'login_attempts' not in session:
        session['login_attempts'] = 0
    
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'account_id' not in session:
        return redirect(url_for('login'))
    
    user_data = session.get('user_data')
    account_data = session.get('account_data')
    
    # Get updated balance from database
    try:
        updated_account = get_user_account(session['account_id'])
        if updated_account:
            session['account_data'] = updated_account
            account_data = updated_account
    except:
        pass
    
    # Reset login attempts when successfully accessing dashboard
    session['login_attempts'] = 1  # Set to 1 since they successfully logged in
    
    return render_template('dashboard.html', 
                         username=f"{user_data['first_name']} {user_data['last_name']}",
                         balance=account_data['account_balance'],
                         account_id=user_data['account_id'],
                         email=user_data['email'])

@app.route('/transaction', methods=['GET', 'POST'])
def transaction():
    if 'account_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            user_data = session.get('user_data')
            account_data = session.get('account_data')
            
            # Get form data
            amount = float(request.form.get('amount'))
            recipient_account = request.form.get('recipient_account')
            message = request.form.get('message', '')
            
            # Refresh account balance from database
            updated_account = get_user_account(session['account_id'])
            if updated_account:
                account_data = updated_account
                session['account_data'] = updated_account
            
            # Validate amount
            if amount <= 0:
                flash('Amount must be greater than 0', 'error')
                return redirect(url_for('transaction'))
            
            # Check sufficient balance
            if amount > account_data['account_balance']:
                flash(f'Insufficient balance. Your available balance is ${account_data["account_balance"]:.2f}', 'error')
                return redirect(url_for('transaction'))
            
            # Validate recipient account format (12 digits)
            if not recipient_account or len(recipient_account) != 12 or not recipient_account.isdigit():
                flash('Invalid recipient account number. Must be exactly 12 digits.', 'error')
                return redirect(url_for('transaction'))
            
            # Check if trying to transfer to own account
            if recipient_account == account_data['nomor_rekening']:
                flash('Cannot transfer to your own account', 'error')
                return redirect(url_for('transaction'))
            
            # Validate recipient account exists and get recipient data
            try:
                # First get the recipient account
                recipient_account_check = supabase.table('rekening').select('*').eq('nomor_rekening', recipient_account).execute()
                if not recipient_account_check.data:
                    flash('Recipient account not found. Please check the account number.', 'error')
                    return redirect(url_for('transaction'))
                recipient_account_data = recipient_account_check.data[0]
                
                # Then get the recipient's personal info from nasabah table
                recipient_user_check = supabase.table('nasabah').select('first_name, last_name').eq('account_id', recipient_account_data['account_id']).execute()
                if not recipient_user_check.data:
                    flash('Recipient information not found.', 'error')
                    return redirect(url_for('transaction'))
                recipient_user_data = recipient_user_check.data[0]
                
            except Exception as e:
                print(f"Error validating recipient: {e}")  # For debugging
                flash('Error validating recipient account. Please try again.', 'error')
                return redirect(url_for('transaction'))
            
            # Store transaction data in session for confirmation page
            session['pending_transaction'] = {
                'amount': amount,
                'recipient_account': recipient_account,
                'message': message,
                'sender_account': account_data['nomor_rekening'],
                'sender_name': f"{user_data['first_name']} {user_data['last_name']}",
                'recipient_name': f"{recipient_user_data['first_name']} {recipient_user_data['last_name']}"
            }
            
            # Redirect to confirmation page instead of processing immediately
            return redirect(url_for('confirm_transaction'))
        
        except ValueError as e:
            flash('Invalid amount. Please enter a valid number.', 'error')
            return redirect(url_for('transaction'))
        except Exception as e:
            print(f"Transaction error: {e}")  # For debugging
            flash(f'Error processing transaction: {str(e)}', 'error')
            return redirect(url_for('transaction'))
    
    # GET request - show transaction form
    user_data = session.get('user_data')
    account_data = session.get('account_data')
    
    # Refresh account balance from database
    try:
        updated_account = get_user_account(session['account_id'])
        if updated_account:
            account_data = updated_account
            session['account_data'] = updated_account
    except:
        pass
    
    return render_template('transaction.html',
                         user_data=user_data,
                         account_data=account_data)

@app.route('/confirm_transaction', methods=['GET', 'POST'])
def confirm_transaction():
    if 'account_id' not in session:
        return redirect(url_for('login'))
    
    # Check if there's pending transaction data
    if 'pending_transaction' not in session:
        flash('No pending transaction found. Please start a new transaction.', 'error')
        return redirect(url_for('transaction'))
    
    if request.method == 'POST':
        try:
            user_data = session.get('user_data')
            account_data = session.get('account_data')
            pending_tx = session.get('pending_transaction')
            
            # Get PIN from form
            entered_pin = request.form.get('pin', '').strip()
            
            # Validate PIN format (basic validation - not empty and numeric)
            if not entered_pin or not entered_pin.isdigit():
                flash('Invalid PIN. Please enter a valid numeric PIN.', 'error')
                return redirect(url_for('confirm_transaction'))
            
            # Verify PIN
            user_pin = user_data.get('pin', '')
            if entered_pin != user_pin:
                flash('Incorrect PIN. Please try again.', 'error')
                return redirect(url_for('confirm_transaction'))
            
            # PIN is correct, proceed with transaction processing using NEW MODEL
            # Refresh account balance one more time
            updated_account = get_user_account(session['account_id'])
            if updated_account:
                account_data = updated_account
                session['account_data'] = updated_account
            
            # Final balance check
            if pending_tx['amount'] > account_data['account_balance']:
                flash('Insufficient balance. Your balance may have changed.', 'error')
                session.pop('pending_transaction', None)
                return redirect(url_for('transaction'))
            
            # Get recipient data for balance update
            try:
                recipient_account_data = supabase.table('rekening').select('*').eq('nomor_rekening', pending_tx['recipient_account']).execute()
                if not recipient_account_data.data:
                    flash('Recipient account no longer exists.', 'error')
                    session.pop('pending_transaction', None)
                    return redirect(url_for('transaction'))
                recipient_data = recipient_account_data.data[0]
            except Exception as e:
                flash('Error validating recipient account.', 'error')
                session.pop('pending_transaction', None)
                return redirect(url_for('transaction'))
            
            # Process with NEW ML model
            prediction, probability, ml_features = process_transaction_new_model(
                amount=pending_tx['amount'],
                sender_account=account_data['nomor_rekening'],
                recipient_account=pending_tx['recipient_account']
            )
            
            # Generate transaction ID
            transaction_id = f"TXN{random.randint(100000, 999999)}"
            
            # Prepare transaction record for database
            transaction_record = {
                'transaction_id': transaction_id,
                'transaction_date': datetime.now().isoformat(),
                'ip_address': request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', '127.0.0.1')),
                'nomor_rekening_pengirim': account_data['nomor_rekening'],
                'nomor_rekening_penerima': pending_tx['recipient_account'],
                'amount': pending_tx['amount'],
                'message': pending_tx['message']
            }
            
            # Save transaction to database
            save_transaction_to_db(transaction_record)
            
            if prediction == 0:  # Not fraud - process immediately
                try:
                    # Update sender balance
                    new_sender_balance = account_data['account_balance'] - pending_tx['amount']
                    update_account_balance(account_data['nomor_rekening'], new_sender_balance)
                    
                    # Update recipient balance
                    new_recipient_balance = recipient_account_data.data[0]['account_balance'] + pending_tx['amount']
                    update_account_balance(pending_tx['recipient_account'], new_recipient_balance)
                    
                    # Update session balance
                    session['account_data']['account_balance'] = new_sender_balance
                    session['balance'] = new_sender_balance
                    
                    # Get current user data for result display
                    current_user = session.get('user_data')
                    current_account = session.get('account_data')
                    
                    # Clear pending transaction
                    session.pop('pending_transaction', None)
                    
                    flash(f'Transaction completed successfully! ${pending_tx["amount"]:.2f} sent to {pending_tx["recipient_account"]}', 'success')
                    return render_template('result.html', 
                                         result=int(prediction), 
                                         probability=probability,
                                         amount=pending_tx['amount'],
                                         transaction_id=transaction_id,
                                         recipient_account=pending_tx['recipient_account'],
                                         recipient_name=pending_tx['recipient_name'],
                                         sender_account=current_account['nomor_rekening'],
                                         sender_name=f"{current_user['first_name']} {current_user['last_name']}",
                                         message=pending_tx.get('message'),
                                         transaction_date=datetime.now(),
                                         status='success',
                                         ml_features=ml_features)
                except Exception as e:
                    flash(f'Error processing transaction: {str(e)}', 'error')
                    return redirect(url_for('transaction'))
                    
            else:  # Potential fraud - redirect to fraud detection page
                # Store fraud detection data in session
                session['fraud_detection'] = {
                    'prediction': int(prediction),
                    'probability': probability,
                    'amount': pending_tx['amount'],
                    'recipient_account': pending_tx['recipient_account'],
                    'recipient_name': pending_tx['recipient_name'],
                    'transaction_id': transaction_id,
                    'transaction_time': datetime.now().strftime('%H:%M:%S'),
                    'ml_features': ml_features
                }
                
                # Clear pending transaction since we're moving to fraud flow
                session.pop('pending_transaction', None)
                
                return redirect(url_for('fraud_detected'))
        
        except Exception as e:
            print(f"Confirm transaction error: {e}")  # For debugging
            flash(f'Error confirming transaction: {str(e)}', 'error')
            return redirect(url_for('confirm_transaction'))
    
    # GET request - show confirmation page
    pending_tx = session.get('pending_transaction')
    
    return render_template('confirm_transaction.html',
                         transaction_data=pending_tx)

@app.route('/fraud_detected')
def fraud_detected():
    if 'account_id' not in session:
        return redirect(url_for('login'))
    
    # Check if there's fraud detection data
    if 'fraud_detection' not in session:
        flash('No fraud detection data found.', 'error')
        return redirect(url_for('dashboard'))
    
    fraud_data = session.get('fraud_detection')
    
    return render_template('fraud.html',
                         probability=fraud_data.get('probability'),
                         amount=fraud_data.get('amount'),
                         recipient_account=fraud_data.get('recipient_account'),
                         transaction_time=fraud_data.get('transaction_time'),
                         transaction_id=fraud_data.get('transaction_id'),
                         ml_features=fraud_data.get('ml_features', {}))

@app.route('/send_otp', methods=['POST'])
def send_otp():
    if 'account_id' not in session:
        return redirect(url_for('login'))
    
    if 'fraud_detection' not in session:
        flash('No fraud detection data found.', 'error')
        return redirect(url_for('dashboard'))
    
    try:
        user_data = session.get('user_data')
        fraud_data = session.get('fraud_detection')
        
        # Generate 4-digit OTP seperti di test_email.py
        otp = ''.join(str(random.randint(0, 9)) for _ in range(4))
        print(f"Generated OTP: {otp}")
        
        # Store OTP and related data in session
        session['pending_otp'] = otp
        session['pending_amount'] = fraud_data['amount']
        session['pending_recipient'] = fraud_data['recipient_account']
        session['pending_recipient_name'] = fraud_data['recipient_name']
        session['transaction_id'] = fraud_data['transaction_id']
        
        # Send OTP via email
        user_email = user_data.get('email', '')
        user_name = f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}"
        
        email_sent = send_email_otp(user_email, otp, user_name)
        
        if email_sent:
            flash(f'OTP telah dikirim ke email {user_email}. Silakan cek inbox Anda.', 'info')
        else:
            flash(f'Gagal mengirim email. Untuk testing, OTP Anda adalah: {otp}', 'warning')
        
        # Clear fraud detection data
        session.pop('fraud_detection', None)
        
        return redirect(url_for('enter_otp'))
        
    except Exception as e:
        print(f"Error dalam send_otp: {e}")
        flash(f'Error sending OTP: {str(e)}', 'error')
        return redirect(url_for('fraud_detected'))

@app.route('/cancel_transaction', methods=['POST'])
def cancel_transaction():
    if 'account_id' not in session:
        return redirect(url_for('login'))
    
    # Clear all pending transaction data
    session.pop('pending_transaction', None)
    session.pop('fraud_detection', None)
    session.pop('pending_otp', None)
    session.pop('transaction_id', None)
    session.pop('pending_amount', None)
    session.pop('pending_recipient', None)
    session.pop('pending_recipient_name', None)
    session.pop('pending_message', None)
    
    flash('Transaksi telah dibatalkan.', 'info')
    return redirect(url_for('dashboard'))

@app.route('/resend_otp')
def resend_otp():
    if 'account_id' not in session:
        return redirect(url_for('login'))
    
    if 'pending_otp' not in session:
        flash('No pending OTP found.', 'error')
        return redirect(url_for('dashboard'))
    
    try:
        user_data = session.get('user_data')
        
        # Generate new 4-digit OTP seperti di test_email.py
        otp = ''.join(str(random.randint(0, 9)) for _ in range(4))
        session['pending_otp'] = otp
        print(f"Regenerated OTP: {otp}")
        
        # Send OTP via email
        user_email = user_data.get('email', '')
        user_name = f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}"
        
        email_sent = send_email_otp(user_email, otp, user_name)
        
        if email_sent:
            flash(f'OTP baru telah dikirim ke email {user_email}. Silakan cek inbox Anda.', 'info')
        else:
            flash(f'Gagal mengirim email. Untuk testing, OTP baru Anda adalah: {otp}', 'warning')
        
        return redirect(url_for('enter_otp'))
        
    except Exception as e:
        print(f"Error dalam resend_otp: {e}")
        flash(f'Error resending OTP: {str(e)}', 'error')
        return redirect(url_for('enter_otp'))

@app.route('/enter_otp', methods=['GET', 'POST'])
def enter_otp():
    if 'account_id' not in session or 'pending_otp' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        user_input = request.form.get('otp', '').strip()
        
        # Validate OTP format (4 digits)
        if not user_input or len(user_input) != 4 or not user_input.isdigit():
            flash('Format OTP tidak valid. Masukkan 4 digit angka.', 'error')
            return redirect(url_for('enter_otp'))
        
        if user_input == session.get('pending_otp'):
            try:
                # Process the verified transaction
                account_data = session.get('account_data')
                amount = session.get('pending_amount')
                recipient_account = session.get('pending_recipient')
                recipient_name = session.get('pending_recipient_name')
                transaction_id = session.get('transaction_id')
                
                # Refresh account balance
                updated_account = get_user_account(session['account_id'])
                if updated_account:
                    account_data = updated_account
                    session['account_data'] = updated_account
                
                # Final balance check
                if amount > account_data['account_balance']:
                    flash('Saldo tidak mencukupi. Saldo Anda mungkin telah berubah.', 'error')
                    # Clear pending data
                    session.pop('pending_otp', None)
                    session.pop('transaction_id', None)
                    session.pop('pending_amount', None)
                    session.pop('pending_recipient', None)
                    session.pop('pending_recipient_name', None)
                    session.pop('pending_message', None)
                    return redirect(url_for('transaction'))
                
                # Update sender balance
                new_balance = account_data['account_balance'] - amount
                update_account_balance(account_data['nomor_rekening'], new_balance)
                
                # Update recipient balance
                recipient_data = supabase.table('rekening').select('*').eq('nomor_rekening', recipient_account).execute().data[0]
                new_recipient_balance = recipient_data['account_balance'] + amount
                update_account_balance(recipient_account, new_recipient_balance)
                
                # Update session balance
                session['account_data']['account_balance'] = new_balance
                session['balance'] = new_balance
                
                # Get current user and account data for result display
                current_user = session.get('user_data')
                current_account = session.get('account_data')
                
                # Store result data temporarily
                result_data = {
                    'amount': amount,
                    'transaction_id': transaction_id,
                    'recipient_account': recipient_account,
                    'recipient_name': recipient_name,
                    'sender_account': current_account['nomor_rekening'],
                    'sender_name': f"{current_user['first_name']} {current_user['last_name']}",
                    'message': session.get('pending_message', ''),
                    'transaction_date': datetime.now()
                }
                
                # Clear pending transaction data
                session.pop('pending_otp', None)
                session.pop('transaction_id', None)
                session.pop('pending_amount', None)
                session.pop('pending_recipient', None)
                session.pop('pending_recipient_name', None)
                session.pop('pending_message', None)
                
                flash(f"OTP berhasil diverifikasi. Transaksi selesai! ${amount:.2f} telah dikirim ke {recipient_account}", "success")
                
                # Redirect to result page with success
                return render_template('result.html', 
                                     result=0,  # Success
                                     probability=0,
                                     amount=result_data['amount'],
                                     transaction_id=result_data['transaction_id'],
                                     recipient_account=result_data['recipient_account'],
                                     recipient_name=result_data['recipient_name'],
                                     sender_account=result_data['sender_account'],
                                     sender_name=result_data['sender_name'],
                                     message=result_data['message'],
                                     transaction_date=result_data['transaction_date'],
                                     status='success')
                
            except Exception as e:
                flash(f"Error menyelesaikan transaksi: {str(e)}", "error")
                return redirect(url_for('transaction'))
        else:
            flash("OTP salah. Transaksi dibatalkan.", "error")
            # Clear pending transaction data
            session.pop('pending_otp', None)
            session.pop('transaction_id', None)
            session.pop('pending_amount', None)
            session.pop('pending_recipient', None)
            session.pop('pending_recipient_name', None)
            session.pop('pending_message', None)
            return redirect(url_for('dashboard'))
    
    # GET request - show OTP input page
    user_data = session.get('user_data')
    user_email = user_data.get('email', '') if user_data else ''
    
    return render_template('enter_otp.html', user_email=user_email)

@app.route('/history')
def history():
    if 'account_id' not in session:
        return redirect(url_for('login'))
    
    try:
        account_data = session.get('account_data')
        user_account = account_data['nomor_rekening']
        
        # Get transactions where user is sender or recipient
        response = supabase.table('transaction').select('*').or_(
            f'nomor_rekening_pengirim.eq.{user_account},nomor_rekening_penerima.eq.{user_account}'
        ).order('transaction_date', desc=True).execute()
        
        transactions_data = response.data
        
        # Enhance transactions with recipient names
        enhanced_transactions = []
        for transaction in transactions_data:
            try:
                # Parse transaction_date if it's a string
                if isinstance(transaction.get('transaction_date'), str):
                    from datetime import datetime
                    transaction['transaction_date'] = datetime.fromisoformat(transaction['transaction_date'].replace('Z', '+00:00'))
                
                # Get recipient name if current user is sender
                if transaction['nomor_rekening_pengirim'] == user_account:
                    # User is sender, get recipient info
                    recipient_account_data = supabase.table('rekening').select('account_id').eq('nomor_rekening', transaction['nomor_rekening_penerima']).execute()
                    if recipient_account_data.data:
                        recipient_user = supabase.table('nasabah').select('first_name, last_name').eq('account_id', recipient_account_data.data[0]['account_id']).execute()
                        if recipient_user.data:
                            transaction['recipient_name'] = f"{recipient_user.data[0]['first_name']} {recipient_user.data[0]['last_name']}"
                        else:
                            transaction['recipient_name'] = 'Unknown Recipient'
                    else:
                        transaction['recipient_name'] = 'Unknown Recipient'
                else:
                    # User is recipient, get sender info
                    sender_account_data = supabase.table('rekening').select('account_id').eq('nomor_rekening', transaction['nomor_rekening_pengirim']).execute()
                    if sender_account_data.data:
                        sender_user = supabase.table('nasabah').select('first_name, last_name').eq('account_id', sender_account_data.data[0]['account_id']).execute()
                        if sender_user.data:
                            transaction['sender_name'] = f"{sender_user.data[0]['first_name']} {sender_user.data[0]['last_name']}"
                        else:
                            transaction['sender_name'] = 'Unknown Sender'
                    else:
                        transaction['sender_name'] = 'Unknown Sender'
                
                enhanced_transactions.append(transaction)
                
            except Exception as e:
                print(f"Error processing transaction {transaction.get('transaction_id', 'unknown')}: {e}")
                # Add transaction even if we can't get names
                enhanced_transactions.append(transaction)
        
        return render_template('history.html', 
                             transactions=enhanced_transactions, 
                             user_account=user_account)
        
    except Exception as e:
        print(f"Error loading transaction history: {e}")
        flash(f"Error loading transaction history: {str(e)}", "error")
        return render_template('history.html', transactions=[], user_account="")

@app.route('/transaction_detail/<transaction_id>')
def transaction_detail(transaction_id):
    """Show detailed view of a specific transaction"""
    if 'account_id' not in session:
        return redirect(url_for('login'))
    
    try:
        account_data = session.get('account_data')
        user_data = session.get('user_data')
        user_account = account_data['nomor_rekening']
        
        # Get specific transaction
        response = supabase.table('transaction').select('*').eq('transaction_id', transaction_id).execute()
        
        if not response.data:
            flash('Transaction not found.', 'error')
            return redirect(url_for('history'))
        
        transaction = response.data[0]
        
        # Verify user has access to this transaction
        if transaction['nomor_rekening_pengirim'] != user_account and transaction['nomor_rekening_penerima'] != user_account:
            flash('Access denied to this transaction.', 'error')
            return redirect(url_for('history'))
        
        # Parse transaction_date if it's a string
        if isinstance(transaction.get('transaction_date'), str):
            from datetime import datetime
            transaction_date = datetime.fromisoformat(transaction['transaction_date'].replace('Z', '+00:00'))
        else:
            transaction_date = transaction.get('transaction_date')
        
        # Determine sender and recipient info
        if transaction['nomor_rekening_pengirim'] == user_account:
            # Current user is sender
            sender_name = f"{user_data['first_name']} {user_data['last_name']}"
            sender_account = user_account
            
            # Get recipient info
            recipient_account_data = supabase.table('rekening').select('account_id').eq('nomor_rekening', transaction['nomor_rekening_penerima']).execute()
            if recipient_account_data.data:
                recipient_user = supabase.table('nasabah').select('first_name, last_name').eq('account_id', recipient_account_data.data[0]['account_id']).execute()
                if recipient_user.data:
                    recipient_name = f"{recipient_user.data[0]['first_name']} {recipient_user.data[0]['last_name']}"
                else:
                    recipient_name = 'Unknown Recipient'
            else:
                recipient_name = 'Unknown Recipient'
            recipient_account = transaction['nomor_rekening_penerima']
        else:
            # Current user is recipient
            recipient_name = f"{user_data['first_name']} {user_data['last_name']}"
            recipient_account = user_account
            
            # Get sender info
            sender_account_data = supabase.table('rekening').select('account_id').eq('nomor_rekening', transaction['nomor_rekening_pengirim']).execute()
            if sender_account_data.data:
                sender_user = supabase.table('nasabah').select('first_name, last_name').eq('account_id', sender_account_data.data[0]['account_id']).execute()
                if sender_user.data:
                    sender_name = f"{sender_user.data[0]['first_name']} {sender_user.data[0]['last_name']}"
                else:
                    sender_name = 'Unknown Sender'
            else:
                sender_name = 'Unknown Sender'
            sender_account = transaction['nomor_rekening_pengirim']
        
        return render_template('result.html',
                             result=0,  # Completed transaction
                             probability=0,
                             amount=transaction['amount'],
                             transaction_id=transaction['transaction_id'],
                             recipient_account=recipient_account,
                             recipient_name=recipient_name,
                             sender_account=sender_account,
                             sender_name=sender_name,
                             message=transaction.get('message'),
                             transaction_date=transaction_date,
                             status='completed')
        
    except Exception as e:
        print(f"Error loading transaction detail: {e}")
        flash(f"Error loading transaction detail: {str(e)}", "error")
        return redirect(url_for('history'))

@app.route('/info')
def info():
    if 'account_id' not in session:
        return redirect(url_for('login'))
    
    user_data = session.get('user_data')
    account_data = session.get('account_data')
    
    # Get updated balance from database
    try:
        updated_account = get_user_account(session['account_id'])
        if updated_account:
            session['account_data'] = updated_account
            account_data = updated_account
    except:
        pass
    
    # Format birth date for display
    formatted_birth_date = 'Not set'
    if user_data and user_data.get('tanggal_lahir'):
        try:
            # If it's a string, parse it
            if isinstance(user_data['tanggal_lahir'], str):
                birth_date = datetime.strptime(user_data['tanggal_lahir'], '%Y-%m-%d')
            else:
                birth_date = user_data['tanggal_lahir']
            formatted_birth_date = birth_date.strftime('%B %d, %Y')
        except (ValueError, AttributeError):
            formatted_birth_date = user_data['tanggal_lahir']  # Show as is if parsing fails
    
    # Define list of available locations
    locations = [
        'Jakarta',
        'Surabaya',
        'Bandung',
        'Medan',
        'Semarang',
        'Makassar',
        'Palembang',
        'Tangerang',
        'Depok',
        'Bekasi',
        'Bogor',
        'Batam',
        'Pekanbaru',
        'Bandar Lampung',
        'Malang',
        'Padang',
        'Denpasar',
        'Samarinda',
        'Tasikmalaya',
        'Balikpapan',
        'Pontianak',
        'Manado',
        'Yogyakarta',
        'Banjarmasin',
        'Jambi',
        'Cirebon',
        'Mataram',
        'Kendari',
        'Palu',
        'Ambon'
    ]
    
    return render_template('info.html', 
                         user_data=user_data, 
                         account_data=account_data,
                         formatted_birth_date=formatted_birth_date,
                         locations=locations)

@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'account_id' not in session:
        return redirect(url_for('login'))
    
    try:
        account_id = session['account_id']
        
        # Get form data
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        email = request.form.get('email', '').strip()
        tanggal_lahir = request.form.get('tanggal_lahir', '').strip()
        customer_occupation = request.form.get('customer_occupation', '').strip()
        location = request.form.get('location', '').strip()
        
        # Validate required fields
        if not all([first_name, last_name, email, tanggal_lahir, customer_occupation, location]):
            flash('All fields are required', 'error')
            return redirect(url_for('info'))
        
        # Validate date format
        try:
            datetime.strptime(tanggal_lahir, '%Y-%m-%d')
        except ValueError:
            flash('Invalid date format', 'error')
            return redirect(url_for('info'))
        
        # Validate email format (basic)
        if '@' not in email or '.' not in email:
            flash('Invalid email format', 'error')
            return redirect(url_for('info'))
        
        # Update user data in Supabase
        update_data = {
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'tanggal_lahir': tanggal_lahir,
            'customer_occupation': customer_occupation,
            'location': location
        }
        
        response = supabase.table('nasabah').update(update_data).eq('account_id', account_id).execute()
        
        if response.data:
            # Update session data
            session['user_data'].update(update_data)
            flash('Profile updated successfully!', 'success')
        else:
            flash('Failed to update profile. Please try again.', 'error')
    
    except Exception as e:
        print(f"Error updating profile: {e}")  # For debugging
        flash(f'Error updating profile: {str(e)}', 'error')
    
    return redirect(url_for('info'))

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for('login'))

# API endpoints for testing
@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy", "database": "supabase"})

@app.route('/api/test_db')
def test_db():
    try:
        response = supabase.table('nasabah').select('account_id').limit(1).execute()
        return jsonify({"status": "success", "data": response.data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/test_model')
def test_model():
    """Test endpoint untuk model baru dengan 7 variabel dan threshold 0.8"""
    try:
        # Test dengan data dummy
        test_amount = 1000000
        test_sender = "123456789012"
        test_recipient = "987654321098"
        
        prediction, probability, features = process_transaction_new_model(
            amount=test_amount,
            sender_account=test_sender,
            recipient_account=test_recipient
        )
        
        return jsonify({
            "status": "success",
            "model_loaded": model is not None,
            "scaler_loaded": scaler is not None,
            "fraud_threshold": 0,
            "prediction": int(prediction),
            "prediction_label": "FRAUD" if prediction == 1 else "LEGITIMATE",
            "fraud_probability": probability,
            "features_used": {
                "amount": features.get('amount'),
                "oldBalInitiator": features.get('oldBalInitiator'),
                "newBalInitiator": features.get('newBalInitiator'),
                "oldBalRecipient": features.get('oldBalRecipient'),
                "newBalRecipient": features.get('newBalRecipient'),
                "isFlaggedFraud": features.get('isFlaggedFraud'),
                "waktu": features.get('waktu')
            },
            "time_category": get_time_category(),
            "current_time": datetime.now().strftime('%H:%M:%S'),
            "note": "Model menggunakan 7 variabel dengan threshold 0.8 untuk deteksi fraud"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)