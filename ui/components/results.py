import streamlit as st

def render_results(results):
    st.subheader("📄 Results")

    for i, r in enumerate(results, 1):
        st.markdown(f"""
        ### {i}. Document {r['doc_id']}
        **Score:** {r['score']:.4f}

        {r['snippet']}
        ---
        """)

# """
# Module Name: results.py
# Purpose: Component to render retrieval results with dynamic topic tag highlights.
# """
# import streamlit as st
#
# # قاموس لربط أسماء المواضيع دلالياً (بناءً على التوزيع الذي ظهر في التدريب)
# TOPIC_NAMES = {
#     0: "General & Sentiment",
#     1: "Business & Platforms",
#     2: "Life & Global Perspectives",
#     3: "Social & Relations",
#     4: "Technology & Software Apps",
#     5: "Human Behavior & Identity",
#     6: "E-Commerce & Websites",
#     7: "Finance & Numbers",
#     8: "Education & Communication",
#     9: "Engineering & Mathematics"
# }
#
# # قاموس للألوان المقابلة لكل موضوع لتظهر كـ Tags جذابة
# TOPIC_COLORS = {
#     0: "#8884d8", 1: "#82ca9d", 2: "#ffc658", 3: "#ff7300", 4: "#0088FE",
#     5: "#00C49F", 6: "#FFBB28", 7: "#FF8042", 8: "#a4de6c", 9: "#d0ed57"
# }
#
#
# def render_results(results):
#     st.subheader("📄 Results")
#
#     for i, r in enumerate(results, 1):
#         doc_id = r.get('doc_id')
#         score = r.get('score', 0.0)
#         snippet = r.get('snippet', 'No content available.')
#         st.markdown(f"**Document ID:** `{doc_id}`")
#         topic_id = r.get('detected_topic_id', None)  # القيمة القادمة من الباكيند
#
#         # 1. عرض العنوان الأساسي والـ Score بناءً على كودك
#         st.markdown(f"### {i}. Document {doc_id}")
#         st.markdown(f"**Score:** {score:.4f}")
#
#         # 2. ✨ [حقن الـ Tag الملون]: إذا كانت ميزة الـ Topic Pruning نشطة وحسبت الموضوع
#         if topic_id is not None and topic_id in TOPIC_NAMES:
#             t_name = TOPIC_NAMES[topic_id]
#             t_color = TOPIC_COLORS.get(topic_id, "#777777")
#             badge_html = f"""
#             <span style='background-color: {t_color}; color: white; padding: 3px 10px;
#                          border-radius: 4px; font-size: 13px; font-weight: bold; margin-bottom: 8px; display: inline-block;'>
#                 📌 Topic {topic_id}: {t_name}
#             </span>
#             """
#             st.markdown(badge_html, unsafe_allow_html=True)
#
#         # 3. عرض الـ snippet المقتبس والخط الفاصل من كودك الأصلي
#         st.markdown(f"""
#         {snippet}
#         ---
#         """)