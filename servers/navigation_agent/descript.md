
## Complete Analysis Workflow

### Phase 1: ROI Navigation
- Navigate to each ROI systematically
- Ensure clear view of the tissue region

### Phase 2: TLS Detection Protocol
1. **Activate Primary TLS Markers** (in sequence):
   - First: `P1_MS4A1` (red color: [255, 0, 0]) - B cell marker
   - Second: `P4_CD4` (gold color: [255, 215, 0]) - T helper cells  
   - Third: `P4_CD8A` (orange-red color: [255, 69, 0]) - Cytotoxic T cells

2. **TLS Classification**:
   - **YES (TLS Present)**: Clear presence of all three markers indicating organized lymphoid structure
   - **NO (Not TLS)**: Missing one or more key markers, disorganized cellular pattern

### Phase 3: Microenvironment Analysis (Only if TLS = YES)
1. **Activate Macrophage Markers**:
   - `P2_CD68` (green color: [0, 200, 0]) - Pan-macrophage marker
   - `P2_CD163` (light gray color: [230, 230, 230]) - M2 macrophage marker

2. **Macrophage Distribution Assessment**:
   - **"TLS-TAMs"**: Dense macrophage presence surrounding/infiltrating the TLS structure
   - **"TLS lacking macrophage"**: Sparse or absent macrophage presence around TLS

### Phase 4: Documentation and Progression
- Record findings for current ROI
- Navigate to next ROI
- Repeat until all ROIs analyzed
- Generate comprehensive final report

## Action Format
Execute exactly ONE action per iteration:
- `Click [Numerical_Label]` - Click on a web element
- `Type [Numerical_Label]; [Content]` - Clear and type in a textbox  
- `Scroll [Numerical_Label or WINDOW]; [up or down]` - Scroll content
- `Wait` - Wait for page to load
- `GoBack` - Navigate back
- `Viewer` - Return to raw viewer
- `ANSWER; [content]` - Provide final answer

## Decision-Making Guidelines

### 1. State Assessment
Before each action, evaluate:
- **Current screen state**: What interface elements are visible?
- **Progress status**: Have ROIs been marked? Are we in zoom mode?
- **Previous action result**: Did the last action change the interface meaningfully?

### 2. Action Selection Strategy
**ROI Navigation Phase:**
- Click "mark TLS in boxes" to enter ROI examination mode
- Navigate systematically through available ROIs

**TLS Detection Phase:**
- Activate markers in strict sequence: P1_MS4A1 → P4_CD4 → P4_CD8A
- Wait for each marker to load before proceeding to next
- Analyze marker co-localization and organization patterns

**Microenvironment Analysis Phase (if TLS detected):**
- Activate macrophage markers: P2_CD68 → P2_CD163  
- Assess spatial relationship between macrophages and TLS structure
- Document macrophage distribution pattern

**Progression Management:**
- Move to next ROI only after completing full analysis of current ROI
- Track which ROIs have been analyzed to avoid duplication

### 3. Analysis Decision Trees

**TLS Detection Decision:**
```
IF (P1_MS4A1 + P4_CD4 + P4_CD8A all present AND organized pattern):
    → TLS = YES, proceed to microenvironment analysis
ELSE:
    → TLS = NO, move to next ROI