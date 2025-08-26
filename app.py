import streamlit as st

page_main = st.Page("app.py", title="Main Page", icon="👉")
p1 = st.Page("user_input.py",title="Page 1",icon="🔧")
p2 = st.Page("p1.py", title="Page 2",icon="🔧")


page=st.navigation([p1,p2])
page.run()
