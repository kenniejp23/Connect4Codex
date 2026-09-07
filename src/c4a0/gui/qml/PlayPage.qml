import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: page
    objectName: "playPage"
    property bool analysisVisible: true
    property bool editingText: ApplicationWindow.window.activeFocusItem &&
        (ApplicationWindow.window.activeFocusItem.hasOwnProperty("text") ||
         ApplicationWindow.window.activeFocusItem.hasOwnProperty("editable"))
    property bool compact: width < 950

    Shortcut { sequence: "1"; enabled: page.visible && !page.editingText && Game.humanTurn; onActivated: Game.makeMove(0) }
    Shortcut { sequence: "2"; enabled: page.visible && !page.editingText && Game.humanTurn; onActivated: Game.makeMove(1) }
    Shortcut { sequence: "3"; enabled: page.visible && !page.editingText && Game.humanTurn; onActivated: Game.makeMove(2) }
    Shortcut { sequence: "4"; enabled: page.visible && !page.editingText && Game.humanTurn; onActivated: Game.makeMove(3) }
    Shortcut { sequence: "5"; enabled: page.visible && !page.editingText && Game.humanTurn; onActivated: Game.makeMove(4) }
    Shortcut { sequence: "6"; enabled: page.visible && !page.editingText && Game.humanTurn; onActivated: Game.makeMove(5) }
    Shortcut { sequence: "7"; enabled: page.visible && !page.editingText && Game.humanTurn; onActivated: Game.makeMove(6) }
    Shortcut { sequence: "U"; enabled: page.visible && !page.editingText && Game.active; onActivated: Game.undo() }
    Shortcut { sequence: "N"; enabled: page.visible && !page.editingText && Game.active; onActivated: Game.rematch() }
    Shortcut { sequence: "B"; enabled: page.visible && !page.editingText && Game.active; onActivated: Game.bestMove() }
    Shortcut { sequence: "R"; enabled: page.visible && !page.editingText && Game.active; onActivated: Game.randomMove() }

    ColumnLayout {
        anchors.fill: parent
        anchors.leftMargin: 28
        anchors.rightMargin: 28
        anchors.bottomMargin: 24
        spacing: 14

        Panel {
            Layout.fillWidth: true
            Layout.preferredHeight: page.compact ? 178 : 104
            GridLayout {
                columns: page.compact ? 3 : 6
                uniformCellWidths: true
                anchors.fill: parent
                anchors.margins: 14
                columnSpacing: page.compact ? 8 : 14
                rowSpacing: 8

                Column {
                    Layout.fillWidth: true; Layout.minimumWidth: 140
                    spacing: 6
                    Text { text: "RED PLAYER"; color: ApplicationWindow.window.dangerColor; font.pixelSize: 10; font.weight: Font.DemiBold; font.letterSpacing: 1 }
                    ComboBox { id: redPlayer; property string selectedPlayer: "Human"; onActivated: selectedPlayer = currentText; onModelChanged: Qt.callLater(function() { currentIndex = Math.max(0, find(selectedPlayer)) }); width: parent.width; height: 42; model: App.playerOptions; currentIndex: 0 }
                }
                Column {
                    Layout.fillWidth: true; Layout.minimumWidth: 140
                    spacing: 6
                    Text { text: "GOLD PLAYER"; color: ApplicationWindow.window.accentColor; font.pixelSize: 10; font.weight: Font.DemiBold; font.letterSpacing: 1 }
                    ComboBox { id: goldPlayer; property string selectedPlayer: App.preferredPlayPlayer; onActivated: selectedPlayer = currentText; onModelChanged: Qt.callLater(function() { currentIndex = Math.max(0, find(selectedPlayer)) }); width: parent.width; height: 42; model: App.playerOptions; currentIndex: Math.max(0, find(App.preferredPlayPlayer)) }
                }
                Column {
                    Layout.fillWidth: true; Layout.minimumWidth: 140
                    spacing: 6
                    Text { text: "MCTS ITERATIONS"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 9; font.weight: Font.DemiBold }
                    SpinBox { id: gameIterations; width: parent.width; height: 42; from: 1; to: 100000; value: 1400; editable: true }
                }
                Column {
                    Layout.fillWidth: true
                    spacing: 6
                    Text { text: "EXPLORATION"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 9; font.weight: Font.DemiBold }
                    TextField { id: gameExploration; width: parent.width; height: 42; text: "6.6"; validator: DoubleValidator { bottom: 0 } }
                }
                Column {
                    Layout.fillWidth: true
                    spacing: 6
                    Text { text: "PLY PENALTY"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 9; font.weight: Font.DemiBold }
                    TextField { id: gamePly; width: parent.width; height: 42; text: "0.01"; validator: DoubleValidator { bottom: 0 } }
                }
                PrimaryButton {
                    objectName: "startGameButton"
                    Layout.fillWidth: true
                    Layout.preferredHeight: 48
                    text: Game.active ? (page.compact ? "Restart" : "Restart setup") : (page.compact ? "Start" : "Start game")

                    onClicked: Game.startGame(redPlayer.currentText, goldPlayer.currentText,
                                              gameIterations.value, Number(gameExploration.text), Number(gamePly.text))
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 14

            Panel {
                id: boardPanel
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.minimumWidth: page.compact ? 440 : 490
                Layout.maximumWidth: page.compact && page.analysisVisible ? Math.max(440, page.width - 330) : 16777215
                color: ApplicationWindow.window.dark ? "#0E1829" : "#F8FAFC"

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 10

                    RowLayout {
                        Layout.fillWidth: true
                        Text {
                            text: Game.status
                            Accessible.role: Accessible.StaticText
                            Accessible.name: text
                            onTextChanged: function() {
                                if (page.visible && Game.active) Accessible.announce(Game.status, Accessible.Polite)
                            }
                            color: Game.terminalState === "red_win" ? ApplicationWindow.window.dangerColor
                                 : (Game.terminalState === "gold_win" ? ApplicationWindow.window.accentColor : ApplicationWindow.window.textColor)
                            font.pixelSize: 18
                            font.weight: Font.DemiBold
                        }
                        Item { Layout.fillWidth: true }
                        Text { visible: !page.compact; text: Game.moveCount + " moves"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 12 }
                        ToolButton { text: page.compact ? "Analysis" : (page.analysisVisible ? "Hide analysis" : "Show analysis"); onClicked: page.analysisVisible = !page.analysisVisible }
                    }

                    Item {
                        id: boardArea
                        objectName: "boardArea"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        property real boardWidth: Math.min(width, (height - 42) * 7 / 6)
                        property real cellSize: boardWidth / 7

                        Item {
                            id: boardFrame
                            objectName: "boardFrame"
                            width: boardArea.boardWidth
                            height: boardArea.cellSize * 6 + 42
                            anchors.centerIn: parent

                            Item {
                                id: previewRow
                                width: parent.width
                                height: 38
                                property int hoverColumn: -1
                                Rectangle {
                                    width: Math.min(32, boardArea.cellSize * 0.66)
                                    height: width
                                    radius: width / 2
                                    x: previewRow.hoverColumn * boardArea.cellSize + (boardArea.cellSize - width) / 2
                                    anchors.verticalCenter: parent.verticalCenter
                                    visible: previewRow.hoverColumn >= 0 && Game.humanTurn && Game.legalMoves[previewRow.hoverColumn]
                                    color: Game.sideToMove === "red" ? "#EF5B67AA" : "#F4C44EAA"
                                    border.width: 2
                                    border.color: "#FFFFFF88"
                                }
                            }

                            Rectangle {
                                id: board
                                objectName: "gameBoard"
                                anchors.top: previewRow.bottom
                                width: parent.width
                                height: boardArea.cellSize * 6
                                radius: 18
                                color: "#15549A"
                                border.width: Game.terminalState === "draw" ? 4 : 2
                                border.color: Game.terminalState === "draw" ? ApplicationWindow.window.accentColor : "#3374B8"

                                Grid {
                                    id: boardGrid
                                    anchors.fill: parent
                                    property real gridMargin: Math.max(5, boardArea.cellSize * 0.08)
                                    anchors.margins: gridMargin
                                    rows: 6
                                    columns: 7
                                    property real gridSpacing: Math.max(4, boardArea.cellSize * 0.05)
                                    spacing: gridSpacing
                                    Repeater {
                                        model: 42
                                        Item {
                                            id: cell
                                            required property int index
                                            property int piece: Game.board[index]
                                            property bool winning: {
                                                var revision = Game.revision
                                                return Game.isWinningCell(index)
                                            }
                                            width: (board.width - boardGrid.gridMargin * 2 - boardGrid.gridSpacing * 6) / 7
                                            height: (board.height - boardGrid.gridMargin * 2 - boardGrid.gridSpacing * 5) / 6

                                            Rectangle {
                                                anchors.centerIn: parent
                                                width: Math.min(parent.width, parent.height) * 0.82
                                                height: width
                                                radius: width / 2
                                                color: cell.piece === 1 ? "#EF5B67" : (cell.piece === 2 ? "#F4C44E" : "#0A2341")
                                                border.width: cell.winning ? 4 : 2
                                                border.color: cell.winning ? "#FFFFFF" : (cell.piece === 0 ? "#0A1C35" : "#FFFFFF44")
                                                transform: [
                                                    Translate { id: dropTranslation; y: 0 },
                                                    Scale { id: tokenScale; origin.x: width / 2; origin.y: height / 2; xScale: 1; yScale: 1 }
                                                ]

                                                onColorChanged: {
                                                    if (cell.piece > 0 && !App.reducedMotion)
                                                        dropAnimation.restart()
                                                }
                                                SequentialAnimation {
                                                    id: dropAnimation
                                                    ParallelAnimation {
                                                        NumberAnimation { target: dropTranslation; property: "y"; from: -(Math.floor(cell.index / 7) + 1) * boardArea.cellSize; to: 0; duration: 260; easing.type: Easing.InCubic }
                                                        NumberAnimation { target: tokenScale; property: "yScale"; from: 0.84; to: 1.08; duration: 260; easing.type: Easing.OutCubic }
                                                    }
                                                    NumberAnimation { target: tokenScale; property: "yScale"; to: 1.0; duration: 90; easing.type: Easing.OutQuad }
                                                }
                                                SequentialAnimation on opacity {
                                                    running: cell.winning && !App.reducedMotion
                                                    loops: 3
                                                    NumberAnimation { from: 1; to: 0.38; duration: 300 }
                                                    NumberAnimation { from: 0.38; to: 1; duration: 300 }
                                                }
                                            }
                                        }
                                    }
                                }

                                Row {
                                    anchors.fill: parent
                                    Repeater {
                                        model: 7
                                        MouseArea {
                                            required property int index
                                            width: board.width / 7
                                            height: board.height
                                            activeFocusOnTab: true
                                            Accessible.role: Accessible.Button
                                            Accessible.name: "Drop in column " + (index + 1)
                                            Accessible.description: Game.status + ". Column cells top to bottom: " + Game.board.filter(function(_, i) { return i % 7 === index }).map(function(piece) { return piece === 0 ? "empty" : (piece === 1 ? "red" : "gold") }).join(", ")
                                            Accessible.onPressAction: if (enabled) Game.makeMove(index)
                                            Keys.onReturnPressed: if (enabled) Game.makeMove(index)
                                            Keys.onSpacePressed: if (enabled) Game.makeMove(index)
                                            Rectangle { anchors.fill: parent; color: "transparent"; border.width: parent.activeFocus ? 3 : 0; border.color: "white" }
                                            hoverEnabled: true
                                            enabled: page.visible && !page.editingText && Game.humanTurn && Game.legalMoves[index]
                                            cursorShape: enabled ? Qt.PointingHandCursor : Qt.ForbiddenCursor
                                            onEntered: previewRow.hoverColumn = index
                                            onExited: if (previewRow.hoverColumn === index) previewRow.hoverColumn = -1
                                            onClicked: { forceActiveFocus(); Game.makeMove(index) }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: page.compact ? 4 : 8
                        Button { Layout.preferredWidth: page.compact ? 52 : implicitWidth; Layout.minimumWidth: page.compact ? 48 : implicitWidth; leftPadding: page.compact ? 8 : 24; rightPadding: page.compact ? 8 : 24; text: page.compact ? "↶" : "Undo  U"; ToolTip.visible: hovered; ToolTip.text: "Undo (U)"; enabled: page.visible && !page.editingText && Game.active && Game.moveCount > 0; onClicked: Game.undo() }
                        Button { Layout.preferredWidth: page.compact ? 52 : implicitWidth; Layout.minimumWidth: page.compact ? 48 : implicitWidth; leftPadding: page.compact ? 8 : 24; rightPadding: page.compact ? 8 : 24; text: page.compact ? "N" : "New  N"; ToolTip.visible: hovered; ToolTip.text: "New game (N)"; enabled: page.visible && !page.editingText && Game.active; onClicked: Game.rematch() }
                        Button { Layout.preferredWidth: page.compact ? 52 : implicitWidth; Layout.minimumWidth: page.compact ? 48 : implicitWidth; leftPadding: page.compact ? 8 : 24; rightPadding: page.compact ? 8 : 24; text: page.compact ? "B" : "Best  B"; ToolTip.visible: hovered; ToolTip.text: "Best move (B)"; enabled: page.visible && !page.editingText && Game.active && Game.terminalState === "ongoing"; onClicked: Game.bestMove() }
                        Button { Layout.preferredWidth: page.compact ? 52 : implicitWidth; Layout.minimumWidth: page.compact ? 48 : implicitWidth; leftPadding: page.compact ? 8 : 24; rightPadding: page.compact ? 8 : 24; text: page.compact ? "R" : "Random  R"; ToolTip.visible: hovered; ToolTip.text: "Random move (R)"; enabled: page.visible && !page.editingText && Game.active && Game.terminalState === "ongoing"; onClicked: Game.randomMove() }
                        Item { Layout.fillWidth: true }
                        Button { Layout.preferredWidth: page.compact ? 56 : implicitWidth; Layout.minimumWidth: page.compact ? 48 : implicitWidth; leftPadding: page.compact ? 8 : 24; rightPadding: page.compact ? 8 : 24; text: page.compact ? "+1" : "+1 MCTS"; ToolTip.visible: hovered; ToolTip.text: "Add 1 MCTS iteration"; enabled: page.visible && !page.editingText && Game.active; onClicked: Game.addIterations(1) }
                        Button { Layout.preferredWidth: page.compact ? 64 : implicitWidth; Layout.minimumWidth: page.compact ? 52 : implicitWidth; leftPadding: page.compact ? 8 : 24; rightPadding: page.compact ? 8 : 24; text: "+100"; ToolTip.visible: hovered; ToolTip.text: "Add 100 MCTS iterations"; enabled: page.visible && !page.editingText && Game.active; onClicked: Game.addIterations(100) }
                    }
                }

                Rectangle {
                    visible: Game.terminalState !== "ongoing"
                    anchors.horizontalCenter: parent.horizontalCenter
                    anchors.bottom: parent.bottom
                    anchors.bottomMargin: 72
                    width: Math.min(470, parent.width - 40)
                    height: 76
                    radius: 14
                    color: ApplicationWindow.window.panelRaisedColor
                    border.color: ApplicationWindow.window.borderColor
                    RowLayout {
                        anchors.fill: parent; anchors.margins: 12
                        ColumnLayout {
                            Layout.fillWidth: true
                            Text { text: Game.status; color: ApplicationWindow.window.textColor; font.pixelSize: 16; font.weight: Font.Bold }
                            Text { text: Game.moveCount + " moves completed"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                        }
                        PrimaryButton { text: "Rematch"; onClicked: Game.rematch() }
                        Button { text: "Swap"; onClicked: Game.swapSides() }
                        Button { text: "Setup"; onClicked: Game.endGame() }
                    }
                }
            }

            Panel {
                objectName: "analysisPanel"
                visible: page.analysisVisible
                Layout.preferredWidth: page.compact ? 260 : 290
                Layout.fillHeight: true
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 12
                    Text { text: "LIVE ANALYSIS"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 10; font.weight: Font.DemiBold; font.letterSpacing: 1.1 }
                    RowLayout {
                        Layout.fillWidth: true
                        Text { text: Game.searching ? "Searching" : "Search ready"; color: Game.searching ? ApplicationWindow.window.successColor : ApplicationWindow.window.mutedTextColor; font.pixelSize: 12; font.weight: Font.DemiBold }
                        Item { Layout.fillWidth: true }
                        Text { text: Game.iterations + " / " + Game.maxIterations; color: ApplicationWindow.window.textColor; font.pixelSize: 12 }
                    }
                    ProgressBar { Layout.fillWidth: true; from: 0; to: Math.max(1, Game.maxIterations); value: Game.iterations }
                    Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: ApplicationWindow.window.borderColor }
                    GridLayout {
                        Layout.fillWidth: true
                        columns: 2
                        uniformCellWidths: true
                        columnSpacing: 16
                        ColumnLayout {
                            Layout.fillWidth: true
                            Text { text: "EVALUATION"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 9 }
                            Text { text: Game.qPenalty.toFixed(2); color: Game.qPenalty >= 0 ? "#EF5B67" : ApplicationWindow.window.accentColor; font.pixelSize: 24; font.weight: Font.DemiBold }
                        }
                        ColumnLayout {
                            Layout.fillWidth: true
                            Text { text: "EXPECTED OUTCOME"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 9 }
                            Text { text: (Game.qNoPenalty >= 0 ? "+" : "") + Game.qNoPenalty.toFixed(2); color: Game.qNoPenalty >= 0 ? "#EF5B67" : ApplicationWindow.window.accentColor; font.pixelSize: 24; font.weight: Font.DemiBold }
                        }
                    }
                    Text { text: "POLICY BY COLUMN"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 9; font.weight: Font.DemiBold }
                    Repeater {
                        model: 7
                        RowLayout {
                            required property int index
                            Layout.fillWidth: true
                            Text { text: index + 1; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11; Layout.preferredWidth: 16 }
                            ProgressBar { Layout.fillWidth: true; from: 0; to: 1; value: Math.max(0, Game.policy[index]) }
                            Text { text: (Math.max(0, Game.policy[index]) * 100).toFixed(0) + "%"; color: ApplicationWindow.window.textColor; font.pixelSize: 10; Layout.preferredWidth: 34; horizontalAlignment: Text.AlignRight }
                        }
                    }
                    Item { Layout.fillHeight: true }
                    Rectangle {
                        visible: Game.error.length > 0
                        Layout.fillWidth: true
                        Layout.preferredHeight: errorContent.implicitHeight + 24
                        radius: 10
                        color: "#4A1F2A"
                        ColumnLayout {
                            id: errorContent
                            anchors.fill: parent
                            anchors.margins: 12
                            Text { Layout.fillWidth: true; text: Game.error; color: "#FFB3BB"; wrapMode: Text.WordWrap; font.pixelSize: 11 }
                            RowLayout {
                                Button { text: "Retry"; onClicked: Game.retry() }
                                Button { text: "Change model"; flat: true; onClicked: Game.endGame() }
                            }
                        }
                    }
                    Text { text: "Positive values favour Red\nNegative values favour Gold"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 10; lineHeight: 1.3 }
                }
            }
        }
    }
}
