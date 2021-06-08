/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
*
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory).
* Copyright of the model is maintained by the developers, and the model is distributed under
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
*
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer.
*
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* Developer list:
*   Pai-Yu Chen	    Email: pchen72 at asu dot edu
*
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#ifndef NEW_SUBARRAY_H_
#define NEW_SUBARRAY_H_

#include <vector>
#include "typedef.h"
#include "InputParameter.h"
#include "Technology.h"
#include "MemCell.h"
#include "formula.h"
#include "FunctionUnit.h"
#include "Adder.h"
#include "RowDecoder.h"
#include "Mux.h"
#include "WLDecoderOutput.h"
#include "DFF.h"
#include "DeMux.h"
#include "Precharger.h"
#include "SenseAmp.h"
#include "DecoderDriver.h"
#include "SRAMWriteDriver.h"
#include "ReadCircuit.h"
#include "SwitchMatrix.h"
#include "ShiftAdd.h"
#include "WLNewDecoderDriver.h"
#include "NewSwitchMatrix.h"
#include "CurrentSenseAmp.h"
#include "MultilevelSenseAmp.h"
#include "MultilevelSAEncoder.h"
#include "SarADC.h"
#include "SubArray.h"
#include "LevelShifter.h"

using namespace std;

class NewSubArray: public FunctionUnit {
private:
    void BigArrayInitialize();
public:
	NewSubArray(InputParameter& _inputParameter, Technology& _tech, MemCell& _cell);
	virtual ~NewSubArray() {}
	InputParameter& inputParameter;
	Technology& tech;
	MemCell& cell;

	/* Functions */
	void PrintProperty();
	void Initialize(int _numRow, int _numCol, double _unitWireRes, int _subSubArrayRow, int _subSubArrayCol);
	void CalculateArea();
	void CalculateLatency(double _rampInput, const vector<double> &columnResistance, bool CalculateclkFreq);
	void CalculatePower(const vector<double> &columnResistance);
    void PrintId();

	/* Properties */
	bool initialized;	   // Initialization flag
	int numRow;			   // Number of rows
	int numCol;			   // Number of columns
    double unitWireRes;	// Unit wire resistance, Unit ohm/m
	int subSubArrayRow, subSubArrayCol;
	int numSubSubArrayRow, numSubSubArrayCol;

    double heightArray;
	double widthArray;
	double areaArray;
	double topPeripheralLatency, botPeripheralLatency, leftPeripheralLatency, arrayLatency;
	double topPeripheralLeakage, botPeripheralLeakage, leftPeripheralLeakage;
	double arrayLeakage;
	double topPeripheralReadDynamicEnergy, botPeripheralReadDynamicEnergy, leftPeripheralReadDynamicEnergy;
	double arrayReadDynamicEnergy, arrayWriteDynamicEnergy;
	double readDynamicEnergyArray, writeDynamicEnergyArray;
    vector<bool> activeSubArray;

    // From initial SubArray
	int numColMuxed;
	int numWriteColMuxed;
	int totalNumWritePulse;
	int numWritePulseAVG;
	double writeLatencyArray;

	double lengthRow;
	double lengthCol;
	double capRow1;
	double capRow2;
	double capCol;
	double resRow;
	double resCol;
	double resCellAccess;
	double capCellAccess;
	double colDelay;

	double activityRowWrite;
	double activityColWrite;
	double activityRowRead, activityColRead;
	int numReadPulse;
	double numWritePulse;
	int maxNumWritePulse;
	int maxNumIntBit;

	bool neuro;
	bool neuroSimReadSimulation;
	bool multifunctional;
	bool conventionalSequential;
	bool conventionalParallel;
	bool BNNsequentialMode;
	bool BNNparallelMode;
	bool XNORsequentialMode;
	bool XNORparallelMode;
	bool SARADC;
	bool currentMode;
	bool validated;

	int levelOutput;

	ReadCircuitMode readCircuitMode;
	int numWriteCellPerOperationFPGA;
	int numWriteCellPerOperationMemory;
	int numWriteCellPerOperationNeuro;
	double clkFreq;
	int avgWeightBit;
	int numCellPerSynapse;
	int numReadCellPerOperationFPGA;
	int numReadCellPerOperationMemory;
	int numReadCellPerOperationNeuro;
	bool parallelWrite;
	bool FPGA;
	bool LUT_dynamic;
	bool backToBack;
	int numLut;

	int numReadLutPerOperationFPGA;
	SpikingMode spikingMode;

	bool shiftAddEnable;

	bool relaxArrayCellHeight;
	bool relaxArrayCellWidth;

	double areaADC, areaAccum, areaOther, readLatencyADC, readLatencyAccum, readLatencyOther, readDynamicEnergyADC, readDynamicEnergyAccum, readDynamicEnergyOther;
	double readDynamicEnergyStorage;

	bool trainingEstimation, parallelTrans;
	int levelOutputTrans, numRowMuxedTrans, numReadPulseTrans;

	/* Circuit Modules */
    SubArray                *bigArray;   // for ADC, Accum, Other data;
    vector<SubArray>        subSubArray; // Flatten 2D Array
};

#endif /* NEW_SUBARRAY_H_ */
