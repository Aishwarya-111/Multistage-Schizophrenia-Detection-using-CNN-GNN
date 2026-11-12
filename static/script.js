/* ============================================================
   SchizoDetect — script.js
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {

    // ── TAB MANAGEMENT ──────────────────────────────────────
    const navBtns  = document.querySelectorAll('.nav-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const tabHeading    = document.getElementById('tab-heading');
    const tabSubheading = document.getElementById('tab-subheading');

    const TAB_META = {
        'home-tab':    ['Overview', 'Multimodal AI-driven schizophrenia staging platform'],
        'analyze-tab': ['Patient Diagnostic Lab', 'Upload NIfTI scans, rate SANS/SAPS symptoms, and predict schizophrenia stage'],
        'graph-tab':   ['Population Graph', 'Interactive phenotypic similarity network — cohort + uploaded patient']
    };

    function switchTab(tabId) {
        tabPanes.forEach(p => p.classList.remove('active'));
        navBtns.forEach(b => b.classList.remove('active'));
        const pane = document.getElementById(tabId);
        if (pane) pane.classList.add('active');
        navBtns.forEach(b => { if (b.dataset.tab === tabId) b.classList.add('active'); });
        if (TAB_META[tabId]) {
            if (tabHeading)    tabHeading.textContent    = TAB_META[tabId][0];
            if (tabSubheading) tabSubheading.textContent = TAB_META[tabId][1];
        }
        if (tabId === 'graph-tab') loadCohortGraph();
        closeSidebar();
    }

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // ── MOBILE SIDEBAR TOGGLE ────────────────────────────────
    const hamburger = document.getElementById('hamburger-btn');
    const sidebar   = document.getElementById('sidebar-nav');
    const overlay   = document.getElementById('sidebar-overlay');

    function closeSidebar() {
        sidebar  && sidebar.classList.remove('open');
        overlay  && overlay.classList.remove('open');
    }

    if (hamburger) {
        hamburger.addEventListener('click', () => {
            sidebar && sidebar.classList.toggle('open');
            overlay && overlay.classList.toggle('open');
        });
    }

    if (overlay) overlay.addEventListener('click', closeSidebar);

    // ── CLINICAL SCORE CALCULATOR ────────────────────────────
    const sliders       = document.querySelectorAll('#analyze-form input[type="range"]');
    const totalScoreEl  = document.getElementById('saps-sans-total-score');
    const normScoreEl   = document.getElementById('saps-sans-norm-score');

    function recalcScore() {
        let total = 0;
        sliders.forEach(s => { total += parseInt(s.value) || 0; });
        if (totalScoreEl) totalScoreEl.textContent = `${total} / 50`;
        if (normScoreEl)  normScoreEl.textContent  = `${((total / 50) * 100).toFixed(2)}%`;
    }

    sliders.forEach(s => s.addEventListener('input', recalcScore));
    recalcScore();

    // ── FILE UPLOAD DROP ZONE ────────────────────────────────
    const dropZone    = document.getElementById('drop-zone-container');
    const fileInput   = document.getElementById('nifti-file');
    const fileDisplay = document.getElementById('selected-file-name');

    if (dropZone) {
        dropZone.addEventListener('click', () => fileInput && fileInput.click());
        dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('hover'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('hover'));
        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            dropZone.classList.remove('hover');
            if (e.dataTransfer.files[0]) assignFile(e.dataTransfer.files[0]);
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', e => {
            if (e.target.files[0]) assignFile(e.target.files[0]);
        });
    }

    function assignFile(file) {
        const n = file.name.toLowerCase();
        if (!n.endsWith('.nii') && !n.endsWith('.nii.gz')) {
            alert('Please upload a NIfTI file (.nii or .nii.gz).');
            if (fileInput) fileInput.value = '';
            return;
        }
        try {
            const dt = new DataTransfer();
            dt.items.add(file);
            if (fileInput) fileInput.files = dt.files;
        } catch (_) { /* fallback: file already set via input */ }
        if (fileDisplay) {
            fileDisplay.textContent = file.name;
            fileDisplay.style.color = 'var(--teal)';
        }
    }

    // ── PIPELINE SUBMISSION ──────────────────────────────────
    const analyzeForm     = document.getElementById('analyze-form');
    const pipelineOverlay = document.getElementById('pipeline-overlay');
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const resultsPanel    = document.getElementById('results-panel');

    const steps = {
        upload:     document.getElementById('step-upload'),
        preprocess: document.getElementById('step-preprocess'),
        cnn:        document.getElementById('step-cnn'),
        features:   document.getElementById('step-features'),
        gnn:        document.getElementById('step-gnn'),
        gat:        document.getElementById('step-gat')
    };

    const sleep = ms => new Promise(r => setTimeout(r, ms));

    function setStep(el, state) {
        if (!el) return;
        el.className = 'pipeline-step' + (state ? ' ' + state : '');
        const icon = el.querySelector('.step-icon');
        if (!icon) return;
        if (state === 'active') icon.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
        else if (state === 'done') icon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
        else icon.innerHTML = '<i class="fa-solid fa-circle-notch"></i>';
    }

    function resetSteps() {
        Object.values(steps).forEach(el => setStep(el, ''));
    }

    if (analyzeForm) {
        analyzeForm.addEventListener('submit', async e => {
            e.preventDefault();

            if (!fileInput || !fileInput.files.length) {
                alert('Please select a NIfTI MRI file before running the prediction.');
                return;
            }

            if (pipelineOverlay) pipelineOverlay.classList.remove('hidden');
            resetSteps();

            const formData = new FormData(analyzeForm);

            try {
                setStep(steps.upload, 'active');
                const fetchPromise = fetch('/api/predict', { method: 'POST', body: formData });

                await sleep(1800); setStep(steps.upload, 'done'); setStep(steps.preprocess, 'active');
                await sleep(1800); setStep(steps.preprocess, 'done'); setStep(steps.cnn, 'active');
                await sleep(2000); setStep(steps.cnn, 'done'); setStep(steps.features, 'active');
                await sleep(1400); setStep(steps.features, 'done'); setStep(steps.gnn, 'active');

                const res  = await fetchPromise;
                const data = await res.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    if (pipelineOverlay) pipelineOverlay.classList.add('hidden');
                    return;
                }

                setStep(steps.gnn, 'done'); setStep(steps.gat, 'active');
                await sleep(1200); setStep(steps.gat, 'done');
                await sleep(400);

                if (pipelineOverlay) pipelineOverlay.classList.add('hidden');
                renderReport(data);

            } catch (err) {
                console.error(err);
                alert('Server error. Please check your connection and try again.');
                if (pipelineOverlay) pipelineOverlay.classList.add('hidden');
            }
        });
    }

    // ── RENDER RESULTS ───────────────────────────────────────
    let currentSession = null;
    let currentAxis    = 'axial';
    let currentSlice   = 32;

    function renderReport(data) {
        if (resultsPlaceholder) resultsPlaceholder.classList.add('hidden');
        if (resultsPanel)       resultsPanel.classList.remove('hidden');

        currentSession = data.session_id;

        // Stage banner
        const stage   = data.prediction.predicted_stage;
        const conf    = data.prediction.confidence;
        const banner  = document.getElementById('banner-predicted-stage-box');
        if (banner) {
            banner.className = 'stage-banner';
            const cls = { Healthy:'card-stage-healthy', Early:'card-stage-early', Moderate:'card-stage-moderate', Chronic:'card-stage-chronic' };
            if (cls[stage]) banner.classList.add(cls[stage]);
        }
        setText('results-predicted-stage', stage === 'Healthy' ? 'Healthy Controls' : stage + ' Stage');
        setText('results-confidence-pct', `${conf.toFixed(1)}%`);

        const circle = document.getElementById('results-confidence-circle');
        if (circle) circle.style.strokeDashoffset = String(238 - (238 * conf / 100));

        // Slice viewer
        currentAxis = 'axial'; currentSlice = 32;
        document.querySelectorAll('.axis-btn').forEach(b => b.classList.remove('active'));
        const ab = document.querySelector('.axis-btn[data-axis="axial"]');
        if (ab) ab.classList.add('active');
        const scrub = document.getElementById('slice-scrubber');
        if (scrub) scrub.value = 32;
        preload(currentSession, 'axial');
        updateViewer();

        // Probabilities
        const p = data.prediction.probabilities;
        animBar('prob-healthy-bar',  'prob-healthy-txt',  p.Healthy);
        animBar('prob-early-bar',    'prob-early-txt',    p.Early);
        animBar('prob-moderate-bar', 'prob-moderate-txt', p.Moderate);
        animBar('prob-chronic-bar',  'prob-chronic-txt',  p.Chronic);

        setText('results-cnn-healthy', `${data.patient_info.cnn_binary_healthy_prob.toFixed(1)}%`);
        setText('results-cnn-schz',    `${data.patient_info.cnn_binary_schz_prob.toFixed(1)}%`);

        // Biomarkers
        const f = data.key_structural_features;
        setText('feat-vol-ratio',  `${f.brain_volume_ratio.toFixed(2)}%`);
        setText('feat-asymmetry',  `${f.left_right_mean_diff.toFixed(2)}%`);
        setText('feat-hippo-diff', `${f.hippocampus_asymmetry.toFixed(2)}%`);
        setText('feat-ventricle',  `${f.central_ventricle_ratio.toFixed(2)}%`);

        // Neighbours
        const list = document.getElementById('results-neighbors-list');
        if (list) {
            list.innerHTML = '';
            data.phenotypic_neighbors.forEach(n => {
                const item = document.createElement('div');
                item.className = 'neighbor-item';
                const bc = { Healthy:'badge-healthy', Early:'badge-early', Moderate:'badge-moderate', Chronic:'badge-chronic' };
                item.innerHTML = `
                    <div class="neigh-meta">
                        <h6>${n.participant_id} <span class="stage-badge ${bc[n.stage]||'badge-healthy'}">${n.stage}</span></h6>
                        <span>Age ${n.age} &nbsp;·&nbsp; ${n.sex} &nbsp;·&nbsp; Severity: ${n.norm_score.toFixed(1)}%</span>
                    </div>
                    <div class="neigh-sim">
                        <h5>${n.similarity.toFixed(1)}%</h5>
                        <span>match</span>
                    </div>`;
                item.addEventListener('click', () => {
                    switchTab('graph-tab');
                    setTimeout(() => selectNode(n.participant_id), 700);
                });
                list.appendChild(item);
            });
        }

        cohortGraph = data.graph_data;
    }

    function setText(id, val) {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    }

    function animBar(barId, txtId, pct) {
        const bar = document.getElementById(barId);
        const txt = document.getElementById(txtId);
        if (!bar || !txt) return;
        bar.style.width = '0%';
        txt.textContent = '0%';
        setTimeout(() => { bar.style.width = `${pct}%`; txt.textContent = `${pct.toFixed(1)}%`; }, 100);
    }

    // ── SLICE VIEWER ─────────────────────────────────────────
    const sliceImg  = document.getElementById('slice-viewer-img');
    const sliceScrub = document.getElementById('slice-scrubber');
    const sliceLabel = document.getElementById('slice-type-label');
    const axisBtns   = document.querySelectorAll('.axis-btn');

    function updateViewer() {
        if (!currentSession || !sliceImg) return;
        sliceImg.src = `/static/slices/${currentSession}/${currentAxis}_${currentSlice}.png`;
        if (sliceLabel) sliceLabel.textContent = `${currentAxis.charAt(0).toUpperCase()+currentAxis.slice(1)} Slice #${currentSlice}`;
    }

    if (sliceScrub) {
        sliceScrub.addEventListener('input', e => { currentSlice = +e.target.value; updateViewer(); });
    }

    axisBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            axisBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentAxis = btn.dataset.axis;
            preload(currentSession, currentAxis);
            updateViewer();
        });
    });

    function preload(session, axis) {
        if (!session) return;
        for (let i = 0; i < 64; i++) { const img = new Image(); img.src = `/static/slices/${session}/${axis}_${i}.png`; }
    }

    // ── D3 POPULATION GRAPH ──────────────────────────────────
    let cohortGraph = null;
    const svg = d3.select('#network-graph-svg');

    function getSize() {
        const el = document.querySelector('.graph-container');
        return { w: (el && el.clientWidth > 0)  ? el.clientWidth  : 800,
                 h: (el && el.clientHeight > 0) ? el.clientHeight : 460 };
    }

    async function loadCohortGraph() {
        if (cohortGraph) { drawGraph(cohortGraph); return; }
        try {
            const res  = await fetch('/api/cohort-graph');
            const data = await res.json();
            if (!data.error) { cohortGraph = data; drawGraph(data); }
        } catch (e) { console.error('Graph load failed:', e); }
    }

    function drawGraph(graph) {
        svg.selectAll('*').remove();
        const { w, h } = getSize();
        svg.attr('width', w).attr('height', h);
        const g = svg.append('g');

        svg.call(d3.zoom().scaleExtent([.1,4]).on('zoom', ev => g.attr('transform', ev.transform)));

        const colors = { Healthy: '#16a34a', Early: '#0891b2', Moderate: '#d97706', Chronic: '#dc2626', new_patient: '#7c3aed' };

        const sim = d3.forceSimulation(graph.nodes)
            .force('link',   d3.forceLink(graph.links).id(d => d.id).distance(55))
            .force('charge', d3.forceManyBody().strength(-130))
            .force('center', d3.forceCenter(w/2, h/2))
            .force('collide',d3.forceCollide().radius(13));

        const link = g.append('g').selectAll('line').data(graph.links).join('line')
            .attr('stroke',         d => d.is_new_link ? '#7c3aed' : '#cbd5e1')
            .attr('stroke-width',   d => d.is_new_link ? 1.5 : 1)
            .attr('stroke-opacity', d => d.is_new_link ? .7 : .4);

        const node = g.append('g').selectAll('circle').data(graph.nodes).join('circle')
            .attr('r',            d => d.is_new ? 13 : 8)
            .attr('fill',         d => colors[d.is_new ? 'new_patient' : d.stage] || '#94a3b8')
            .attr('stroke',       '#fff')
            .attr('stroke-width', d => d.is_new ? 2.5 : 1.5)
            .attr('cursor', 'pointer')
            .call(d3.drag()
                .on('start', (ev,d) => { if(!ev.active) sim.alphaTarget(.3).restart(); d.fx=d.x; d.fy=d.y; })
                .on('drag',  (ev,d) => { d.fx=ev.x; d.fy=ev.y; })
                .on('end',   (ev,d) => { if(!ev.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }));

        node.on('click', (ev, d) => {
            ev.stopPropagation();
            showInspector(d);
            node.transition().duration(200).attr('opacity', .25);
            d3.select(ev.currentTarget).transition().duration(200).attr('opacity',1).attr('r', d.is_new?15:11);
            const nbrs = new Set([d.id]);
            link.transition().duration(200)
                .attr('stroke-opacity', l => { const h=l.source.id===d.id||l.target.id===d.id; if(h){nbrs.add(l.source.id);nbrs.add(l.target.id);} return h?.85:.05; })
                .attr('stroke-width',  l => (l.source.id===d.id||l.target.id===d.id)?2:1);
            node.filter(n => nbrs.has(n.id)).transition().duration(200).attr('opacity',1);
        });

        svg.on('click', ev => {
            if ((ev.target.tagName||'').toLowerCase()==='svg') {
                node.transition().duration(200).attr('opacity',1).attr('r', d => d.is_new?13:8);
                link.transition().duration(200).attr('stroke-opacity', d => d.is_new_link?.7:.4).attr('stroke-width', d => d.is_new_link?1.5:1);
                hideInspector();
            }
        });

        sim.on('tick', () => {
            link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
            node.attr('cx',d=>d.x).attr('cy',d=>d.y);
        });
    }

    // ── NODE INSPECTOR ───────────────────────────────────────
    const inspectorEmpty = document.getElementById('inspector-placeholder');
    const inspectorData  = document.getElementById('inspector-content');

    function showInspector(n) {
        if (inspectorEmpty) inspectorEmpty.classList.add('hidden');
        if (inspectorData)  inspectorData.classList.remove('hidden');
        setText('inspect-node-id',       n.is_new ? 'Uploaded Patient' : n.id);
        setText('inspect-node-dataset',  n.is_new ? 'Dynamic Inference' : `${(n.dataset||'').toUpperCase()} Dataset`);
        setText('inspect-node-stage',    n.stage  || '—');
        setText('inspect-node-age',      `${(+(n.age||0)).toFixed(1)} yrs`);
        setText('inspect-node-sex',      n.sex    || '—');
        setText('inspect-node-severity', `${(+(n.norm_score||0)).toFixed(1)}%`);
        setText('inspect-node-prob',     `${((+(n.prob_schizophrenia||0))*100).toFixed(1)}%`);
    }

    function hideInspector() {
        if (inspectorEmpty) inspectorEmpty.classList.remove('hidden');
        if (inspectorData)  inspectorData.classList.add('hidden');
    }

    function selectNode(id) {
        svg.selectAll('circle').each(function(d) {
            if (d && d.id === id) this.dispatchEvent(new MouseEvent('click', { bubbles: true }));
        });
    }

    window.addEventListener('resize', () => { if (cohortGraph) drawGraph(cohortGraph); });

}); // end DOMContentLoaded