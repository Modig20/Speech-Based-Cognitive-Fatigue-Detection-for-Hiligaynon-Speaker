import uuid

import streamlit as st

st.set_page_config(page_title="Participant Data Collection", layout="wide")

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        .wizard-shell {
            background: #FFFFFF;
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 18px;
            padding: 1.5rem;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.04);
        }
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid rgba(37, 99, 235, 0.08);
            border-radius: 18px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.4rem 0.8rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .status-low { background: rgba(16, 185, 129, 0.12); color: #047857; }
        .status-moderate { background: rgba(245, 158, 11, 0.14); color: #b45309; }
        .status-high { background: rgba(239, 68, 68, 0.12); color: #b91c1c; }
        div[data-testid="stFileUploaderDropzone"] { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)


DEFAULTS = {
    "session_id": str(uuid.uuid4()),
    "respondent_id": "",
    "language": "Hiligaynon",
    "current_step": 1,
    "current_task_level": "Easy",
    "recorded_audio_bytes": None,
    "samn_perelli_rating": None,
    "inference_results": {},
    "consent_accepted": False,
    "post_debrief_consent": False,
}


for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


TASK_LEVELS = ["Easy", "Moderate", "Intensive"]
SAMN_SCALE = [1, 2, 3, 4, 5, 6, 7]
SAMN_LABELS = {
    1: "Bugtaw gid / Fully alert, wide awake",
    2: "Buhi kag masaligon, pero indi pa pinakamaayo / Very lively, responsive, but not at peak",
    3: "Maayo ang pamatyag kag medyo presko / Okay, somewhat fresh",
    4: "Medyo kapoy, indi na pareho ka-presko / A little tired, less than fresh",
    5: "Kasarang nga kapoy kag daw naluya / Moderately tired, let down",
    6: "Kapoy gid kag mabudlay magkonsentrar / Extremely tired, very difficult to concentrate",
    7: "Gid-ka-kapoy kag indi na makatrabaho sing epektibo / Completely exhausted, unable to function effectively",
}
TASK_PROMPTS = {
    "Easy": {
        "hiligaynon": "May 8 ka nga mangga. Ginhatagan mo ang imo abyan sang 3. Pila ka mangga ang nabilin sa imo? Ipaathag sing matunog ang kada tikang sang imo pagsolbar.",
        "english": "You have 8 mangoes and give 3 to your friend. How many mangoes do you have left? Explain each step of your reasoning aloud.",
    },
    "Moderate": {
        "hiligaynon": "May 4 ka kaumpok sang lapis, kag may 6 ka lapis sa kada kaumpok. Pila tanan ka lapis? Ipaathag sing matunog kon paano mo nakuha ang sabat.",
        "english": "You have 4 bundles of pencils with 6 pencils in each bundle. How many pencils are there altogether? Explain aloud how you worked it out.",
    },
    "Intensive": {
        "hiligaynon": "May 48 ka lapis nga ginpanagtag sing patas sa 6 ka estudyante. Dayon, ang kada estudyante nakabaton pa sang 2 ka dugang nga lapis. Pila tanan ka lapis ang ginpanagtag? Ipaathag sing matunog ang kada tikang sang imo pagsolbar.",
        "english": "48 pencils are shared equally among 6 students. Then each student receives 2 more pencils. How many pencils are distributed altogether? Explain each step of your reasoning aloud.",
    },
}


def next_step():
    step = st.session_state.current_step
    if step == 1 and not st.session_state.consent_accepted:
        return
    if step == 3 and st.session_state.recorded_audio_bytes is None:
        return
    if step == 4 and st.session_state.samn_perelli_rating is None:
        return
    st.session_state.current_step = min(step + 1, 5)


def previous_step():
    st.session_state.current_step = max(st.session_state.current_step - 1, 1)


def render_consent_step():
    st.header("Step 1 · Informed Consent")
    st.markdown(
        """
        <div class="metric-card">
            <h3>Purpose / Katuyuan</h3>
            <p>This study examines speech and responses during short cognitive tasks. Some specific study details are withheld until the debriefing so they do not influence responses.</p>
            <p>Ginatuon sang sini nga pagtuon ang paghambal kag mga sabat samtang nagahimo sang malip-ot nga mga buluhaton sa panghunahuna. Ang pila ka detalye sang pagtuon ipahibalo pagkatapos sang buluhaton agod indi ini makaapekto sa imo mga sabat.</p>
            <h3>What participation involves / Ano ang pag-apil</h3>
            <p>You will answer screening questions, complete a spoken cognitive task, provide a fatigue rating, and submit a voice recording. Speaking and the task may cause temporary mental effort or tiredness.</p>
            <p>Masabat ka sang mga pamangkot para sa screening, maghimo sang buluhaton sa panghunahuna samtang nagahambal, maghatag sang marka sang kakapoy, kag magrekord sang imo tingog. Mahimo ini magdulot sang temporaryo nga pagpanikasog sang hunahuna ukon kakapoy.</p>
            <h3>Confidentiality / Pagtipig sang kompidensyalidad</h3>
            <p>A voice recording can identify you. In this prototype, uploaded audio and responses are sent to the application server for processing during this session but are not saved to a research database. Before recruitment, the research team must implement and explain the approved access, security, storage, and retention arrangements.</p>
            <p>Mahimo makakilala sang tawo paagi sa iya tingog. Sa sini nga prototype, ginapadala sa application server ang audio kag mga sabat para maproseso samtang aktibo ang sesyon, pero wala ini ginatipigan sa database sang pagtuon. Antes mag-recruit sang mga partisipante, dapat ipatuman kag ipahibalo sang research team ang gin-aprubahan nga mga paagi sa pag-access, seguridad, pagtipig, kag pag-retain sang datos.</p>
            <h3>Voluntary participation and withdrawal / Boluntaryo nga pag-apil kag pag-untat</h3>
            <p>Taking part is voluntary. You may skip a question, stop, or withdraw at any time without penalty or loss of benefits. Ask the research team how to request removal of data already submitted under the approved protocol.</p>
            <p>Boluntaryo ang pag-apil. Mahimo mo laktawan ang pamangkot, mag-untat, ukon magbiya sa pagtuon bisan san-o nga wala sing silot ukon madula nga benepisyo. Pamangkuta ang research team kon paano ipapangayo ang pagtangtang sang datos nga naipasa na suno sa gin-aprubahan nga protocol.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("The Hiligaynon translation is a draft. Have a fluent speaker and the approving ethics committee review it; define server security and data-retention arrangements before recruitment.")
    st.checkbox(
        "Nabasa ko kag naintindihan ang impormasyon; boluntaryo ako nga nagauyon mag-apil. / I have read and understood this information, and I voluntarily agree to participate.",
        key="consent_accepted",
    )

    st.button(
        "Next · Sunod",
        type="primary",
        use_container_width=True,
        disabled=not st.session_state.consent_accepted,
        on_click=next_step,
    )


def render_screening_step():
    st.header("Step 2 · Screening")
    st.write("Please provide your demographics and language information before the speech task begins.")

    with st.form("screening_form"):
        st.text_input(
            "Respondent ID",
            placeholder="WVSU_CS_001",
            key="respondent_id",
        )
        st.text_input("Birthplace", placeholder="Iloilo City", key="birthplace")
        st.text_input("Native Language", placeholder="Hiligaynon", key="native_language")
        st.slider(
            "How often do you speak Hiligaynon daily?",
            min_value=1,
            max_value=5,
            value=3,
            key="hiligaynon_frequency_score",
        )
        st.radio(
            "Preferred language for instructions",
            options=["Hiligaynon", "English"],
            horizontal=True,
            key="language",
        )

        previous_col, next_col = st.columns(2)
        with previous_col:
            st.form_submit_button("Previous · Nagligad", on_click=previous_step, use_container_width=True)
        with next_col:
            st.form_submit_button("Next · Sunod", type="primary", on_click=next_step, use_container_width=True)


def render_task_step():
    st.header("Step 3 · Task Wizard")
    task_level = st.selectbox(
        "Select task level",
        TASK_LEVELS,
        index=TASK_LEVELS.index(st.session_state.current_task_level),
        key="current_task_level",
    )
    prompt = TASK_PROMPTS[task_level]
    st.info("Basaha ang pamangkot kag sabta ini paagi sa paghambal. Ipaathag ang imo panghunahuna sing matunog. / Read the prompt and answer aloud, explaining your reasoning.")
    st.markdown(f"**Hiligaynon prompt:** {prompt['hiligaynon']}")
    st.caption(f"English translation: {prompt['english']}")
    st.caption("Prompt wording and difficulty progression are drafts; confirm them against the approved thesis protocol before participant recruitment.")

    uploaded = st.file_uploader(
        "Record or upload speech audio",
        type=["wav", "mp3", "m4a", "ogg"],
        help="Accepted formats: WAV, MP3, M4A, and OGG.",
    )

    if uploaded is not None:
        st.session_state.recorded_audio_bytes = uploaded.read()
        st.audio(st.session_state.recorded_audio_bytes)
        st.success("Audio captured successfully. Proceed to the fatigue rating step.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("Previous · Nagligad", use_container_width=True, on_click=previous_step)
    with col2:
        st.button(
            "Next · Sunod: fatigue rating",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.recorded_audio_bytes is None,
            on_click=next_step,
        )


def render_rating_step():
    st.header("Step 4 · Samn-Perelli Fatigue Rating")
    st.write("Rate your current fatigue level immediately after the task.")

    rating = st.radio(
        "Pilia ang deskripsyon nga pinakabagay sa imo kakapoy / Choose the description that best matches your fatigue level",
        options=SAMN_SCALE,
        format_func=lambda value: f"{value} · {SAMN_LABELS[value]}",
        horizontal=False,
        index=None if st.session_state.samn_perelli_rating is None else SAMN_SCALE.index(st.session_state.samn_perelli_rating),
        key="samn_perelli_rating",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("Previous · Nagligad", use_container_width=True, on_click=previous_step)
    with col2:
        st.button(
            "Next · Sunod: debriefing",
            type="primary",
            use_container_width=True,
            disabled=rating is None,
            on_click=next_step,
        )


def render_debriefing_step():
    st.header("Step 5 · Debriefing")

    st.success("Thank you for completing the task. / Salamat sa paghuman sang buluhaton.")
    st.markdown(
        """
        <div class="wizard-shell">
            <p><strong>Disclosure / Pagpahayag:</strong> The true target of this study is cognitive fatigue as reflected in speech. The task prompts were used to elicit speech while varying cognitive effort; this specific focus was not fully explained before the task to reduce response bias.</p>
            <p>Ang matuod nga ginatuon sang sini nga pagtuon amo ang kakapoy sang panghunahuna nga makita sa paghambal. Gin-gamit ang mga buluhaton agod makakuha sang mga halimbawa sang paghambal samtang nagabag-o ang panikasog sang panghunahuna; wala ginpaathag sing bug-os ang sini nga tuyo antes sang buluhaton agod malikawan ang pagbag-o sang sabat.</p>
            <p>You may now ask questions or withdraw. Your recording and responses were sent to the application server for processing in this session, but this prototype does not save them to a research database or submit them for analysis.</p>
            <p>Mahimo ka na mamangkot ukon magbiya sa pagtuon. Ginpadala sa application server ang imo recording kag mga sabat agod maproseso sa sini nga sesyon, pero wala ini ginatipigan sang prototype sa database sang pagtuon ukon ginapasa para sa pag-usisa.</p>
            <p><strong>Respondent ID:</strong> {respondent_id}</p>
            <p><strong>Task Level / Antas sang buluhaton:</strong> {task_level}</p>
            <p><strong>Samn-Perelli Rating / Marka sang kakapoy:</strong> {rating}</p>
        </div>
        """.format(
            respondent_id=st.session_state.respondent_id or "Not provided",
            task_level=st.session_state.current_task_level,
            rating=(
                f"{st.session_state.samn_perelli_rating} · "
                f"{SAMN_LABELS[st.session_state.samn_perelli_rating]}"
                if st.session_state.samn_perelli_rating is not None
                else "Not provided"
            ),
        ),
        unsafe_allow_html=True,
    )

    st.checkbox(
        "Pagkatapos mahibaluan ang matuod nga katuyuan, nagauyon ako nga gamiton ang akon datos para sa pagtuon. / After learning the true purpose, I agree that my data may be used for the study.",
        key="post_debrief_consent",
    )
    if st.session_state.post_debrief_consent:
        st.success("Post-debriefing confirmation noted in this active session only; it is not saved to a database. / Nakumpirmar ini sa aktibo nga sesyon lamang; wala ini ginatipigan sa database.")
    else:
        st.info("Your post-debriefing confirmation is not given. / Wala mo pa ginkumpirmar ang paggamit sang datos pagkatapos sang debriefing.")

    st.button("Reset session", type="secondary", on_click=reset_session)


def reset_session():
    for key, value in DEFAULTS.items():
        st.session_state[key] = str(uuid.uuid4()) if key == "session_id" else value


st.title("Participant Data Collection")
st.caption("Study workflow for recording speech, rating fatigue, and completing the research task.")

if st.session_state.current_step == 1:
    render_consent_step()
elif st.session_state.current_step == 2:
    render_screening_step()
elif st.session_state.current_step == 3:
    render_task_step()
elif st.session_state.current_step == 4:
    render_rating_step()
elif st.session_state.current_step == 5:
    render_debriefing_step()
