import streamlit as st
import sqlite3
from fetch_case_data_and_summarize import IKApi, query_ai_model
from streamlit_option_menu import option_menu

ikapi = IKApi(maxpages=5)

# Database setup
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            approved BOOLEAN NOT NULL,
            is_admin BOOLEAN NOT NULL
        )"""
    )
    conn.commit()
    conn.close()

def add_user(username, password, approved=False, is_admin=False):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO users (username, password, approved, is_admin) VALUES (?, ?, ?, ?)",
        (username, password, approved, is_admin),
    )
    conn.commit()
    conn.close()

def get_user(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return user

def get_pending_users():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE approved = 0 AND is_admin = 0")
    pending_users = c.fetchall()
    conn.close()
    return pending_users

def approve_user(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("UPDATE users SET approved = 1 WHERE username = ?", (username,))
    conn.commit()
    conn.close()

# Initialize database
init_db()

if not get_user("admin"):
    add_user("admin", "admin123", approved=True, is_admin=True)

# Session management
if "page" not in st.session_state:
    st.session_state.page = "Login"
if "authentication_status" not in st.session_state:
    st.session_state.authentication_status = False
    st.session_state.username = None
    st.session_state.is_admin = False

# Option menu for navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  
        options=["Login", "Register", "Admin Panel", "Legal Query", "Logout"],  # Menu options
        menu_icon="cast", 
        default_index=0,  
        orientation="vertical",  
    )

# Admin Panel for approving users
def admin_panel():
    st.subheader("Admin Panel")
    if st.session_state.is_admin:
        pending_users = get_pending_users()
        if pending_users:
            st.info("Pending User Approvals:")
            for user in pending_users:
                user = user[0]
                st.write(f"User: {user}")
                if st.button(f"Approve {user}", key=f"approve_{user}"):
                    approve_user(user)
                    st.success(f"Approved user: {user}")
        else:
            st.info("No users pending approval.")
    else:
        st.error("Only admins can access this page.")

# Main app content (query and insights)
def main_app():
    st.title("AI Based Legal Research Assistant")
    st.markdown("This app provides detailed insights from related legal cases from Indian Courts.")

    query = st.text_input("Enter your legal query (e.g., 'road accident cases'):")
    if st.button("Analyze"):
        if not query.strip():
            st.warning("Please enter a valid query.")
        else:
            st.info("Fetching related cases...")
            doc_ids = ikapi.fetch_all_docs(query)

            if not doc_ids:
                st.error("No related documents found for your query.")
            else:
                st.success(f"Found {len(doc_ids)} related documents. Processing summaries...")

                all_summaries = []
                for docid in doc_ids[:2]:  
                    case_details = ikapi.fetch_doc(docid)

                    if not case_details:
                        st.warning(f"Failed to fetch details for document ID: {docid}")
                        continue

                    title = case_details.get("title", "No Title")
                    main_text = case_details.get("doc", "")
                    cleaned_text = ikapi.clean_text(main_text)
                    chunks = list(ikapi.split_text_into_chunks(cleaned_text))
                    summaries = []

                    for chunk in chunks:
                        summary = ikapi.summarize(chunk)
                        if summary:
                            summaries.append(summary)

                    final_summary = " ".join(summaries)
                    all_summaries.append(f"Title: {title}\nSummary: {final_summary}")

                combined_summary = "\n\n".join(all_summaries)
                st.subheader("Summarized Case Details")
                st.text_area("Summaries", combined_summary, height=300)

                st.info("Generating insights from summaries...")
                insights = query_ai_model(query, combined_summary)
                st.subheader("AI Insights and Analysis")
                st.write(insights)

# User registration
def register_user():
    st.subheader("Register")
    new_username = st.text_input("Choose a username")
    new_password = st.text_input("Choose a password", type="password")
    if st.button("Register"):
        if get_user(new_username):
            st.error("Username already exists!")
        elif not new_username.strip() or not new_password.strip():
            st.warning("Username and password cannot be empty!")
        else:
            add_user(new_username, new_password)
            st.success("Registration successful! Wait for admin approval.")

# User login
def login_user():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = get_user(username)
        if user:
            stored_password, approved, is_admin = user[1], user[2], user[3]
            if stored_password == password:
                if approved:
                    st.session_state.authentication_status = True
                    st.session_state.username = username
                    st.session_state.is_admin = bool(is_admin)
                    st.success(f"Welcome, {username}!")
                else:
                    st.error("Your account is not approved yet. Please wait for admin approval.")
            else:
                st.error("Invalid username or password.")
        else:
            st.error("Invalid username or password.")

# Logout function
def logout():
    st.session_state.authentication_status = False
    st.session_state.username = None
    st.session_state.is_admin = False
    st.success("You have been logged out.")

# Page logic
if selected == "Login":
    if not st.session_state.authentication_status:
        login_user()
    else:
        st.warning("You are already logged in!")

elif selected == "Register":
    if not st.session_state.authentication_status:
        register_user()
    else:
        st.warning("You are already logged in!")

elif selected == "Admin Panel":
    if st.session_state.authentication_status and st.session_state.is_admin:
        admin_panel()
    else:
        st.error("You must be an admin to access this page.")

elif selected == "Legal Query":
    if st.session_state.authentication_status:
        main_app()
    else:
        st.error("You must log in to access this page.")

elif selected == "Logout":
    if st.session_state.authentication_status:
        logout()
    else:
        st.warning("You are not logged in!")
